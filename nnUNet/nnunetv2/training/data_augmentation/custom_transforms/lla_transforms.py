"""
SLAug-style local intensity augmentation for nnU-Net (batchgeneratorsv2 transforms).

Implements the LLA (Local Location-scale) *mask* variant of SLAug
(Su et al., "Rethinking Data Augmentation for Single-Source Domain
Generalization in Medical Image Segmentation", AAAI 2023):

  * MaskRegionLocalLocationScaleTransform (LLA, mask mode)
        The segmentation label map partitions each training patch into regions,
        and every region gets its own location-scale intensity transform
        x -> a * x + b (sampled per region, and per channel unless synchronized).

Enabled at training time via --slaug_lla --slaug_lla_mode mask (nnUNetTrainer reads the
corresponding nnUNet_slaug_lla* env vars in get_training_transforms).
"""
from __future__ import annotations

import torch
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class MaskRegionLocalLocationScaleTransform(ImageOnlyTransform):
    """SLAug LLA, original-paper variant: per-(label-)region location-scale.

    Faithful to the original SLAug `Local_Location_Scale_Augmentation(image, mask)`:
    the input segmentation label map partitions the patch into regions, and EACH
    region gets its OWN location-scale transform  x -> a*x + b  (sampled per
    region, and per channel unless synchronized). Region boundaries therefore
    produce hard intensity steps (this is the intended behaviour of the paper).

    The transform reads ``data_dict['segmentation']`` to define the regions but
    NEVER modifies it (ImageOnlyTransform). It must run BEFORE the segmentation
    is converted to region one-hot, i.e. while seg is still an integer label map
    (the default position in nnUNet's training transform list). If no
    segmentation is present it is a no-op.
    """

    def __init__(self,
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 scale_range: tuple = (0.8, 1.2),
                 shift_range: tuple = (-0.2, 0.2)):
        super().__init__()
        self.p_per_channel = float(p_per_channel)
        self.synchronize_channels = synchronize_channels
        self.scale_range = scale_range
        self.shift_range = shift_range

    def get_parameters(self, **data_dict) -> dict:
        img: torch.Tensor = data_dict["image"]
        seg = data_dict.get("segmentation", None)
        c = img.shape[0]
        apply_idx = (torch.rand(c) < self.p_per_channel).nonzero(as_tuple=False).flatten()
        n = apply_idx.numel()
        if n == 0 or seg is None:
            return {"apply_idx": apply_idx, "seg_lab": None, "ab_per_channel": None}

        # squeeze a leading channel dim if present -> spatial label map
        seg_lab = seg[0] if seg.ndim == img.ndim else seg
        seg_lab = seg_lab.long()
        labels = [int(v) for v in torch.unique(seg_lab).tolist()]

        def sample_ab():
            a = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
            b = float(torch.empty(1).uniform_(self.shift_range[0], self.shift_range[1]))
            return a, b

        if self.synchronize_channels:
            shared = {lab: sample_ab() for lab in labels}
            ab_per_channel = [shared] * n
        else:
            ab_per_channel = [{lab: sample_ab() for lab in labels} for _ in range(n)]
        return {"apply_idx": apply_idx, "seg_lab": seg_lab,
                "ab_per_channel": ab_per_channel}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        idx: torch.Tensor = params["apply_idx"]
        seg_lab = params["seg_lab"]
        if idx.numel() == 0 or seg_lab is None:
            return img
        for k in range(idx.numel()):
            ch = int(idx[k])
            x = img[ch]
            mn = x.min()
            mx = x.max()
            rng = (mx - mn)
            if float(rng) <= 1e-8:
                continue
            x01 = (x - mn) / rng
            out = x01.clone()
            for lab, (a, b) in params["ab_per_channel"][k].items():
                region = (seg_lab == lab)
                if region.any():
                    out[region] = a * x01[region] + b
            out.clamp_(0.0, 1.0)
            img[ch] = out * rng + mn
        return img
