"""
Size-based post-processing for BraTS-METS predictions (nnU-Net pipeline).

Removes small connected components to suppress false positives. The volume threshold is
selectable PER CLASS (like the original project's config.POSTPROC_MIN_VOL_MM3); a class
threshold of 0 means "do not filter that class". Voxel volume is derived from each image's
affine, so it works for any spacing.

The component-removal logic is identical to the original postprocess.py:
    min_voxels = max(1, ceil(min_volume_mm3 / voxel_vol))
    labeled, n = ndimage.label(binary)          # scipy default 6-connectivity
    sizes = ndimage.sum(binary, labeled, ...)
    drop components whose size < min_voxels      # strict less-than

Usage:
    # RC(label 4) filtered at 5 mm^3, others untouched:
    python postprocess_5mm.py --input <pred_dir> --output <out_dir> --per-class "4:5"
    # all classes at 5 mm^3:
    python postprocess_5mm.py --input <pred_dir> --output <out_dir> --default-min-vol 5
"""
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib
from scipy import ndimage

FILE_ENDING = ".nii.gz"


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume in mm^3 from an affine matrix."""
    voxel_dims = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(np.prod(voxel_dims))


def remove_small_blobs(mask: np.ndarray, label_val: int,
                       min_volume_mm3: float, voxel_vol: float) -> np.ndarray:
    """Remove connected components of `label_val` whose volume < min_volume_mm3.
    Identical to the original postprocess.py implementation."""
    if min_volume_mm3 <= 0:
        return mask
    binary = (mask == label_val)
    if not binary.any():
        return mask
    min_voxels = max(1, int(np.ceil(min_volume_mm3 / voxel_vol)))
    labeled, n_components = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
    out = mask.copy()
    for comp_idx, size in enumerate(sizes, start=1):
        if size < min_voxels:
            out[labeled == comp_idx] = 0
    return out


def postprocess_file(pred_path: Path, out_path: Path,
                     thresholds: Dict[int, float], default_min_vol: float) -> None:
    """Apply per-class small-blob removal to one prediction file.
    thresholds maps class label -> min volume (mm^3); classes not listed use default_min_vol."""
    img = nib.load(str(pred_path))
    data = np.asarray(img.dataobj).astype(np.int32)
    vvol = voxel_volume_mm3(img.affine)

    result = data.copy()
    labels = [int(v) for v in np.unique(result) if int(v) != 0]  # foreground classes present
    for label_val in labels:
        thr = thresholds.get(label_val, default_min_vol)
        result = remove_small_blobs(result, label_val, thr, vvol)

    out_img = nib.Nifti1Image(result.astype(np.int32), img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def postprocess_folder(input_dir, output_dir,
                       thresholds: Dict[int, float] = None,
                       default_min_vol: float = 0.0) -> None:
    """Post-process all .nii.gz predictions in input_dir (probability maps excluded)."""
    thresholds = thresholds or {}
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(f for f in input_dir.glob(f"*{FILE_ENDING}") if "probs" not in f.name)
    print(f"[postprocess_5mm] {len(pred_files)} files | per-class thresholds(mm^3)="
          f"{thresholds or 'none'} | default={default_min_vol}")
    for idx, pred_path in enumerate(pred_files, 1):
        postprocess_file(pred_path, output_dir / pred_path.name, thresholds, default_min_vol)
        if idx % 50 == 0 or idx == len(pred_files):
            print(f"  processed {idx}/{len(pred_files)}")
    print(f"[postprocess_5mm] done -> {output_dir}")


# Default BraTS-METS label name -> integer value (used to resolve name-based thresholds).
LABELS = {"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def parse_per_class(spec: str, labels: Dict[str, int] = None) -> Dict[int, float]:
    """Parse 'key:mm3,key:mm3' -> {label(int): mm3(float)}.
    Each key may be an integer label ('4') or a label name ('RC')."""
    labels = labels or LABELS
    out = {}
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        key, mm3 = tok.split(":", 1)
        key = key.strip()
        lab = labels[key] if key in labels else int(key)
        out[int(lab)] = float(mm3)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove small components per class (BraTS-METS).")
    p.add_argument("--input", required=True, type=Path, help="Directory with predicted .nii.gz files.")
    p.add_argument("--output", required=True, type=Path, help="Output directory for post-processed files.")
    p.add_argument("--per-class", type=str, default="",
                   help="Per-class thresholds, e.g. 'RC:5' / '4:5' / 'NETC:0,SNFH:0,ET:0,RC:5' "
                        "(label name or int : mm^3).")
    p.add_argument("--default-min-vol", type=float, default=0.0,
                   help="Threshold (mm^3) for classes not listed in --per-class (default: 0 = skip).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    postprocess_folder(args.input, args.output,
                       thresholds=parse_per_class(args.per_class),
                       default_min_vol=args.default_min_vol)
