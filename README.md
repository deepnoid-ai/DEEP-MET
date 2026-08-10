# DEEP-MET

Automated brain-metastasis segmentation framework for our submission to the
**BraTS-MET 2026** challenge (Task 1).

The framework combines two self-configuring medical-imaging pipelines:

- **nnU-Net** — voxel-wise multi-class segmentation of the tumor sub-regions
  (NETC, SNFH, ET) and the resection cavity (RC).
- **nnDetection** — object detection to strengthen small-lesion (metastasis) detection.

## Training vs. inference (role split)

- **Training** is done **directly with nnU-Net and nnDetection** via their native CLI
  (`nnUNetv2_plan_and_preprocess` / `nnUNetv2_train`, and the nnDetection commands).
  Training uses the **modified `nnUNet/`** in this repository (it adds the custom training
  flags used below), so install it in editable mode first: `pip install -e ./nnUNet`.
- **Inference** is done **only through the Docker image** (`Dockerfile` + `run.sh`). The
  container runs stock `nnunetv2==2.8.0` for prediction (the trained weights use the
  standard `ResidualEncoderUNet` architecture, so no custom code is needed at test time)
  and drives the full pipeline: nnU-Net segmentation → RC remap / post-processing / merge
  → nnDetection → box-to-segmentation fusion.

## Repository layout

```
DEEP-MET/
├── nnUNet/                     # modified nnU-Net (vendored; see "Third-party code")
├── nnDetection/                # nnDetection (vendored)
├── src/
│   ├── prepare_input.py        # (inference) route /input into nnU-Net / nnDetection layouts
│   ├── data/                   # (training) dataset preparation
│   │   ├── nnunet/
│   │   │   └── make_rc_binary_dataset.py   # build Dataset002 (RC-only binary)
│   │   └── nndet/
│   │       ├── nnUNet_2_nnDet_ET.py        # nnU-Net labels -> nnDetection (ET target)
│   │       └── split_sri_native.py         # split cases into SRI24 / Native tasks
│   └── postprocess/            # (inference) remap / small-component removal / merge / fusion
├── Dockerfile                  # builds the inference container
├── run.sh                      # container entrypoint (/input -> /output)
├── requirements.txt
└── README.md
```

---

## Training

> Install the modified nnU-Net first: `pip install -e ./nnUNet`
> (the `--tversky_plus`, `--tp_gamma`, `--slaug_lla` flags below exist only in this fork).

### nnU-Net

Two models are trained:

- **Dataset001 — primary model** (all classes, **region-based**): full challenge labels
  1=NETC, 2=SNFH, 3=ET, 4=RC.
- **Dataset002 — RC specialist** (**conventional / binary**): RC vs. everything-else.

#### 1. Prepare `dataset.json`

Both datasets use the standard nnU-Net raw layout
(`imagesTr/<case>_0000..0003.nii.gz`, `labelsTr/<case>.nii.gz`). Channel order here is
`0=T1c, 1=T1n, 2=FLAIR, 3=T2w` (match your own data).

**Dataset001 — all classes, region-based.**
Region-based training requires `labels` to be given as *regions* (lists of label values)
together with `regions_class_order`, which maps each region back to a final label
(painted in listed order; later regions overwrite earlier ones):

```json
{
  "channel_names": { "0": "T1c", "1": "T1n", "2": "FLAIR", "3": "T2w" },
  "labels": {
    "background": 0,
    "whole_tumor": [1, 2, 3],
    "tumor_core":  [1, 3],
    "enhancing_tumor": [3],
    "resection_cavity": [4]
  },
  "regions_class_order": [2, 1, 3, 4],
  "numTraining": <N>,
  "file_ending": ".nii.gz"
}
```

The sigmoid heads learn the overlapping regions, and `regions_class_order` paints them
back to the mutually-exclusive challenge labels (whole_tumor→2, tumor_core→1,
enhancing_tumor→3, resection_cavity→4). Adjust the regions / order to match the label
scheme used for the primary model.

**Dataset002 — RC specialist, binary.**
Build it from Dataset001 with the provided script (keeps only cases containing RC, copies
all image channels, and remaps labels to `{0: background/other, 1: RC}`):

```bash
python src/data/nnunet/make_rc_binary_dataset.py \
    --src-dir <nnUNet_raw>/Dataset001_BraTSMET \
    --dst-dir <nnUNet_raw>/Dataset002_BraTSMETRC
```

Its `dataset.json` is a plain 2-label (non-region) configuration:

```json
{
  "channel_names": { "0": "T1c", "1": "T1n", "2": "FLAIR", "3": "T2w" },
  "labels": { "background": 0, "RC": 1 },
  "numTraining": <N>,
  "file_ending": ".nii.gz"
}
```

#### 2. Plan, preprocess, and train

```bash
# --- Dataset001: primary (all-class, region-based) ---
nnUNetv2_plan_and_preprocess -d 001 -pl nnUNetPlannerResEncXL
nnUNetv2_train 001 3d_fullres all -p nnUNetResEncUNetXLPlans \
    --tversky_plus --tp_gamma 1.0 --slaug_lla

# --- Dataset002: RC specialist (conventional / binary) ---
nnUNetv2_plan_and_preprocess -d 002 -pl nnUNetPlannerResEncXL
nnUNetv2_train 002 3d_fullres all -p nnUNetResEncUNetXLPlans \
    --tversky_plus --tp_gamma 1.0 --slaug_lla
```

Both models use the `nnUNetResEncUNetXLPlans` plan, the `3d_fullres` configuration, and
`fold all` (trained on all data).

### nnDetection

nnDetection is trained as a **small-lesion (ET) detector** that complements the nnU-Net
segmentation. Only the **enhancing tumor (ET, label 3)** is used as the detection target,
and each ET lesion is assigned a class by physical volume: **class 0 = small (<27 mm³)** —
the detector's responsibility — and **class 1 = large (≥27 mm³)**. The 27 mm³ threshold
and the ET-only, 26-connectivity lesion definition match the BraTS-MET evaluation protocol
(LabelGroup `[3]`).

Two independent detection models are trained, one per coordinate space, because registered
and unregistered cases have different geometry:

- **Task002_Native** — cases on their original acquisition grid.
- **Task003_SRI24** — cases registered to the SRI24 atlas (fixed 240×240×155 grid).

Both use the `RetinaUNetV001_D3V001_3d` model. Training uses the **vendored `nnDetection/`**
in this repository (modified with `do_seg=True`), so install it in editable mode first
(this is also what the Dockerfile builds):

```bash
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" pip install -v -e ./nnDetection
```

#### 1. Prepare the detection datasets

Start from the primary nnU-Net raw dataset (`Dataset001_BraTSMET`, channels
`0=T1c, 1=T1n, 2=FLAIR, 3=T2w`) and produce the two nnDetection tasks in two steps.

**a. Split by coordinate space.** `split_sri_native.py` inspects each case's image shape
and routes it to `SRI24` (shape == 240×240×155) or `Native` (everything else), preserving
the `imagesTr/imagesTs/labelsTr/labelsTs` layout:

```bash
python src/data/nndet/split_sri_native.py \
    --dataset_dir <nnUNet_raw>/Dataset001_BraTSMET \
    --output_root <split_root> \
    --mode copy
# -> <split_root>/SRI24/Dataset001_BraTSMET/...
# -> <split_root>/Native/Dataset001_BraTSMET/...
```

**b. Convert each split to nnDetection ET-target format.** `nnUNet_2_nnDet_ET.py` turns the
BraTS segmentations into per-lesion instance masks (+ `<case>.json` class maps) in
nnDetection's `raw_splitted/` layout, writing directly into the detection data folder
(`$det_data`). Run it once per space, naming the tasks so they match `run.sh` and the
Dockerfile weight paths (`Task002_Native`, `Task003_SRI24`):

```bash
# Native -> Task002
python src/data/nndet/nnUNet_2_nnDet_ET.py \
    --nnunet_dataset_dir <split_root>/Native/Dataset001_BraTSMET \
    --nndet_workspace    $det_data \
    --task_name          Task002_Native

# SRI24 -> Task003
python src/data/nndet/nnUNet_2_nnDet_ET.py \
    --nnunet_dataset_dir <split_root>/SRI24/Dataset001_BraTSMET \
    --nndet_workspace    $det_data \
    --task_name          Task003_SRI24
```

Each task ends up at `$det_data/TaskXXX/raw_splitted/{imagesTr,labelsTr,...}` with a
`dataset.json` describing the two classes (`small_ET_lt27`, `large_ET_ge27`) and the four
modalities. (Use `--copy_images` if symlinks are a problem on your filesystem.)

#### 2. Set the environment variables

```bash
export det_data=<detection data folder>      # where Task002/Task003 live
export det_models=<detection models folder>  # where trained weights are written
```

#### 3. Preprocess, train, and consolidate

Run the nnDetection pipeline (`nndet_prep → nndet_unpack → nndet_train → nndet_consolidate`)
for **each** task (`002` and `003`). Here we train a **single fold (fold 0)** rather than the
full 5-fold CV:

```bash
# ---- example for Task002_Native (repeat with 003 for Task003_SRI24) ----

# a. plan + preprocess
nndet_prep 002

# b. unpack the preprocessed data for faster training
nndet_unpack ${det_data}/Task002_Native/preprocessed/D3V001_3d/imagesTr 6

# c. train a single fold (fold 0)
nndet_train 002 -o exp.fold=0

# d. consolidate (single fold) + sweep inference boxes
nndet_consolidate 002 RetinaUNetV001_D3V001_3d --num_folds 1 --sweep_boxes
```

Consolidation produces
`${det_models}/Task002_Native/RetinaUNetV001_D3V001_3d/consolidated/`, which is exactly the
path the Dockerfile copies (`COPY models/nndet/ ...`) and `run.sh` reads at inference time.
Do the same for `Task003_SRI24`.

> At inference, `run.sh` runs `nndet_predict` per task (`--num_tta 4`) and fuses the boxes
> into the segmentation with `boxes_2_seg_4.py`, using `score_thresh=0.15` for Native and
> `0.2` for SRI24.

---

## Inference (Docker)

Inference runs entirely inside the Docker container: input is mounted at `/input`, and the
final masks are written to `/output`.

### Weight layout required at build time (IMPORTANT)

The `Dockerfile` copies the trained weights into the image, so **after training you must
place the weights in a `models/` folder at the repository root, using the exact paths
below, before running `docker build`.** These paths are hard-coded in the Dockerfile
(`COPY models/nnunet/ ...`, `COPY models/nndet/ ...`) and referenced by `run.sh`
(`-d 001`, `-d 002`, `Task002_Native`, `Task003_SRI24`, `RetinaUNetV001_D3V001_3d`); a
mismatch will break the build or the run. `models/` is not tracked in git (weights are
large), so it must be created locally.

```
DEEP-MET/
└── models/
    ├── nnunet/                                   # -> /opt/ml/models/nnunet  (nnUNet_results)
    │   ├── Dataset001_BraTSMET/
    │   │   └── nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
    │   │       ├── dataset.json
    │   │       ├── plans.json
    │   │       └── fold_all/checkpoint_final.pth
    │   └── Dataset002_BraTSMETRC/
    │       └── nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
    │           ├── dataset.json
    │           ├── plans.json
    │           └── fold_all/checkpoint_final.pth
    └── nndet/                                     # -> /opt/models  (det_models)
        ├── Task002_Native/RetinaUNetV001_D3V001_3d/consolidated/
        └── Task003_SRI24/RetinaUNetV001_D3V001_3d/consolidated/
```

### Build and run

```bash
# build (from the repository root, after models/ is populated)
docker build -t deep-met .

# run: mount input (read-only) and output
docker run --gpus all --rm \
    -v /path/to/input:/input:ro \
    -v /path/to/output:/output \
    deep-met
```

The container pipeline (see `run.sh`): `prepare_input` → nnU-Net predict (Dataset001
"all" + Dataset002 "RC") → RC remap + per-model 5 mm³ post-processing → merge →
nnDetection predict → box-to-segmentation fusion → flat copy to `/output`.

---

## Third-party code, licenses, and attribution

This repository **vendors** (includes a copy of) two third-party projects. Both are
distributed under the **Apache License 2.0**. Their original `LICENSE` files are retained
inside `nnUNet/` and `nnDetection/`, and their copyright notices are preserved. In
accordance with the Apache-2.0 terms, we note below that these copies have been
**modified** relative to upstream.

| Component | Upstream | Version (commit) | Modified? | License |
|-----------|----------|------------------|-----------|---------|
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet | `2932ced` | Yes (custom trainer / loss and augmentation flags) | Apache-2.0 |
| nnDetection | https://github.com/MIC-DKFZ/nnDetection | `97a58f31` | Yes (`do_seg=True` and related changes) | Apache-2.0 |

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
(the `src/` code) are released under the same Apache-2.0 license unless stated otherwise.
