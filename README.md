# MET-Framework — BraTS-MET 2026

Automated brain-metastasis MRI segmentation framework for our submission to the
**BraTS-MET 2026** challenge (Task 1).

The framework combines two self-configuring medical-imaging pipelines:

- **nnU-Net** — voxel-wise multi-class segmentation of the tumor sub-regions
  (NETC, SNFH, ET) and the resection cavity (RC).
- **nnDetection** — object detection to strengthen small-lesion (metastasis) detection.

Their outputs are combined by the scripts under `post_processing/` and driven by the
configs/tests in `configs/` and `tests/`.

## Repository layout

```
MET-Framework/
├── nnUNet/            # modified nnU-Net (vendored; see "Third-party code" below)
├── nnDetection/       # nnDetection (vendored)
├── configs/           # inference / post-processing configuration
├── post_processing/   # RC remap, small-component removal, model merging
├── tests/             # end-to-end inference drivers
└── README.md
```

## Third-party code, licenses, and attribution

This repository **vendors** (includes a copy of) two third-party projects. Both are
distributed under the **Apache License 2.0**. Their original `LICENSE` files are retained
inside `nnUNet/` and `nnDetection/`, and their copyright notices are preserved. In
accordance with the Apache-2.0 terms, we note below that these copies have been
**modified** relative to upstream.

| Component | Upstream | Version (commit) | Modified? | License |
|-----------|----------|------------------|-----------|---------|
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet | `2932ced` | Yes (custom trainer / loss and related changes) | Apache-2.0 |
| nnDetection | https://github.com/MIC-DKFZ/nnDetection | `97a58f31` | No (upstream, as-is) | Apache-2.0 |

If you use this repository, please **cite the original works**:

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).
> nnU-Net: a self-configuring method for deep learning-based biomedical image
> segmentation. *Nature Methods, 18*(2), 203–211.

> Baumgartner, M., Jäger, P. F., Isensee, F., & Maier-Hein, K. H. (2021).
> nnDetection: A Self-configuring Method for Medical Object Detection.
> *MICCAI 2021*, 530–539.

Please also cite the BraTS-MET / BraTS challenge references as required by the challenge
organizers.

## License

The vendored `nnUNet/` and `nnDetection/` directories remain under their original
Apache-2.0 licenses (see the `LICENSE` files within each directory). Our own additions
(the `configs/`, `post_processing/`, and `tests/` code) are released under the same
Apache-2.0 license unless stated otherwise.
