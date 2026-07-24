"""
Convert BraTS-MET brain-tumor segmentation labels from nnU-Net format into
nnDetection's instance-detection format. Only the enhancing tumor (ET, label 3)
is treated as the detection target: each ET lesion (found via 26-connectivity
connected components) is split into class 0 (small, <27 mm3) or class 1
(large, >=27 mm3) by physical volume. The script writes per-case instance masks,
JSON class mappings, and a dataset.json, while symlinking (or copying) the
original image files into the nnDetection raw layout.

Input  (existing):
    <nnunet_raw>/Dataset001_BraTSMET/
        imagesTr/<case>_0000.nii.gz     (one file per modality)
        imagesTr/<case>_0001.nii.gz
        imagesTr/<case>_0002.nii.gz
        imagesTr/<case>_0003.nii.gz
        imagesTs/<case>_0000.nii.gz
        ...
        labelsTr/<case>.nii.gz          (BraTS-MET seg: 1=NETC, 2=SNFH, 3=ET, 4=RC)
        labelsTs/<case>.nii.gz
        dataset.json

Output (created by this script):
    <nndet_workspace>/<task_name>/
        raw_splitted/
            imagesTr/<case>_0000.nii.gz       (symlink by default)
            imagesTr/<case>_0001.nii.gz
            ...
            imagesTs/...
            labelsTr/<case>.nii.gz            (instance mask: 1..N per lesion)
                    /<case>.json              ({"instances": {"1": cls, ...}})
            labelsTs/<case>.nii.gz
                    /<case>.json
        dataset.json

Lesion + class mapping (BraTS-MET 2026 evaluation protocol):
    lesion = {3} = ET ONLY.   NETC(1), SNFH(2), RC(4) all excluded.
    26-connectivity CCs on the ET mask alone -> matches the official
    evaluation's ET region (LabelGroup [3]), so a small ET touching a
    large NETC/SNFH/RC is still its own small ET instance (no absorption).
    volume_mm3 <  27  ->  class 0 (small)   detection model's responsibility
    volume_mm3 >= 27  ->  class 1 (large)   segmentation model's responsibility

Usage:
    python nnUNet_2_nnDet_ET.py \
        --nnunet_dataset_dir nnUNet_workspace/nnUNet_raw/Dataset001_BraTSMET \
        --nndet_workspace    nnDetection_workspace \
        --task_name          Task004_BraTSMET_sub_ET
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

try:
    import cc3d
    _HAS_CC3D = True
except ImportError:
    import scipy.ndimage as ndi
    _HAS_CC3D = False


# ---------------------------------------------------------------------------
# Protocol constants (match the evaluation script)
# ---------------------------------------------------------------------------
# ET region is LabelGroup [3] in config_mets.yaml -> use label 3 ONLY.
# NETC(1), SNFH(2), RC(4) are NOT part of the detection target here.
LESION_LABELS: Tuple[int, ...] = (3,)             # ET only
VOLUME_THRESHOLD_MM3: float = 27.0
CLASS_SMALL: int = 0                              # nnDetection is 0-indexed
CLASS_LARGE: int = 1


# ---------------------------------------------------------------------------
# Connected components + per-case conversion
# ---------------------------------------------------------------------------
def cc26(mask: np.ndarray) -> np.ndarray:
    """26-connectivity connected components on a binary mask."""
    mask = mask.astype(np.uint8)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    if _HAS_CC3D:
        return cc3d.connected_components(mask, connectivity=26).astype(np.int32)
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    lab, _ = ndi.label(mask, structure=struct)
    return lab.astype(np.int32)


def convert_case(seg_path: Path, out_label_path: Path, out_json_path: Path
                 ) -> Dict[str, int]:
    """Read one BraTS seg, write nnDetection instance mask + JSON.

    Detection target = ET (label 3) only. Each ET connected component is one
    instance; class is decided by its physical volume:
        volume_mm3 <  27  -> class 0 (small)
        volume_mm3 >= 27  -> class 1 (large)
    """
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    vox_vol_mm3 = sx * sy * sz

    # Lesion mask: ET (label 3) ONLY. Everything else ignored.
    lesion = np.zeros(seg.shape, dtype=bool)
    for c in LESION_LABELS:
        lesion |= (seg == c)

    if not lesion.any():
        _save_nifti(np.zeros(seg.shape, dtype=np.int16), img, out_label_path)
        with open(out_json_path, "w") as f:
            json.dump({"instances": {}}, f, indent=2)
        return {"n_lesions": 0, "n_small": 0, "n_large": 0}

    cc = cc26(lesion)
    instance_lab = np.zeros(seg.shape, dtype=np.int16)
    instances: Dict[str, int] = {}
    n_small = 0
    n_large = 0
    new_id = 0

    ids, counts = np.unique(cc, return_counts=True)
    for cc_id, n_vox in zip(ids, counts):
        if cc_id == 0:
            continue
        vol_mm3 = float(n_vox) * vox_vol_mm3
        new_id += 1
        instance_lab[cc == cc_id] = new_id
        if vol_mm3 >= VOLUME_THRESHOLD_MM3:       # >= 27 -> large (eval def)
            instances[str(new_id)] = CLASS_LARGE
            n_large += 1
        else:                                     # < 27 -> small
            instances[str(new_id)] = CLASS_SMALL
            n_small += 1

    _save_nifti(instance_lab, img, out_label_path)
    with open(out_json_path, "w") as f:
        json.dump({"instances": instances}, f, indent=2)

    return {"n_lesions": new_id, "n_small": n_small, "n_large": n_large}


def _save_nifti(arr: np.ndarray, ref: nib.Nifti1Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ref.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(arr.astype(np.int16), ref.affine, header), str(out))


# ---------------------------------------------------------------------------
# Folder-level helpers
# ---------------------------------------------------------------------------
def convert_split(seg_dir: Path, out_label_dir: Path, split_name: str
                  ) -> Dict[str, int]:
    """Convert every seg in a split folder; return aggregate statistics."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    seg_files = sorted(seg_dir.glob("*.nii.gz"))
    totals = {
        "n_cases": 0, "n_failed": 0, "n_empty": 0,
        "n_with_small": 0, "n_with_large": 0, "n_with_both": 0,
        "n_lesions": 0, "n_small": 0, "n_large": 0,
    }
    for seg_path in tqdm(seg_files, desc=f"Converting {split_name}"):
        case_id = seg_path.name[: -len(".nii.gz")]
        try:
            stat = convert_case(seg_path,
                                out_label_dir / f"{case_id}.nii.gz",
                                out_label_dir / f"{case_id}.json")
        except Exception as e:
            print(f"[{case_id}] FAILED: {e}")
            totals["n_failed"] += 1
            continue
        totals["n_cases"] += 1
        totals["n_lesions"] += stat["n_lesions"]
        totals["n_small"] += stat["n_small"]
        totals["n_large"] += stat["n_large"]
        if stat["n_lesions"] == 0:
            totals["n_empty"] += 1
        if stat["n_small"] > 0:
            totals["n_with_small"] += 1
        if stat["n_large"] > 0:
            totals["n_with_large"] += 1
        if stat["n_small"] > 0 and stat["n_large"] > 0:
            totals["n_with_both"] += 1
    return totals


def link_or_copy_dir(src: Path, dst: Path, copy: bool,
                     suffix: str = ".nii.gz") -> int:
    """Place every <src>/*<suffix> into <dst> as symlink (or copy)."""
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(f"*{suffix}"))
    for f in files:
        target = dst / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy:
            shutil.copy2(f, target)
        else:
            os.symlink(f.resolve(), target)
    return len(files)


def detect_modalities(nnunet_dataset_json: Path) -> Dict[str, str]:
    """Read modality names from an nnU-Net dataset.json if present.
    Supports both v2 ('channel_names') and v1 ('modality')."""
    if not nnunet_dataset_json.is_file():
        return {}
    try:
        with open(nnunet_dataset_json) as f:
            meta = json.load(f)
    except Exception:
        return {}
    src = meta.get("channel_names", meta.get("modality", {}))
    if isinstance(src, dict):
        return {str(k): str(v) for k, v in src.items()}
    return {}


def write_dataset_json(task_dir: Path, task_name: str,
                       modalities: Dict[str, str],
                       has_test_labels: bool) -> Path:
    """Write the nnDetection dataset.json describing classes and modalities."""
    payload = {
        "task": task_name,
        "name": task_name.split("_", 1)[-1] if "_" in task_name else task_name,
        "target_class": None,
        "test_labels": has_test_labels,
        "labels": {
            str(CLASS_SMALL): "small_ET_lt27",
            str(CLASS_LARGE): "large_ET_ge27",
        },
        "modalities": modalities if modalities else {
            "0": "modality_0", "1": "modality_1",
            "2": "modality_2", "3": "modality_3",
        },
        "dim": 3,
    }
    out = task_dir / "dataset.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BraTS-MET seg (nnU-Net layout) -> nnDetection labels (ET only).")
    parser.add_argument("--nnunet_dataset_dir", required=True,
                        help="Path to nnUNet_raw/DatasetXXX_BraTSMET")
    parser.add_argument("--nndet_workspace", required=True,
                        help="Root folder to write the nnDetection task into.")
    parser.add_argument("--task_name", default="Task001_BraTSMET",
                        help="nnDetection task folder name. Default Task001_BraTSMET.")
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy image files instead of symlinking (slower, "
                             "uses more disk).")
    args = parser.parse_args()

    nnunet_dir = Path(args.nnunet_dataset_dir).resolve()
    task_dir = Path(args.nndet_workspace).resolve() / args.task_name
    raw_splitted = task_dir / "raw_splitted"
    raw_splitted.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BraTS-MET seg  ->  nnDetection conversion  (ET-only detection target)")
    print("=" * 72)
    print(f"  input (nnU-Net raw)  : {nnunet_dir}")
    print(f"  output (nnDet task)  : {task_dir}")
    print(f"  lesion labels        : {LESION_LABELS} (ET only; NETC/SNFH/RC excluded)")
    print(f"  volume threshold     : {VOLUME_THRESHOLD_MM3} mm^3")
    print(f"  class mapping        : <27 -> {CLASS_SMALL} (small), "
          f">=27 -> {CLASS_LARGE} (large)")
    print(f"  cc3d available       : {_HAS_CC3D}")
    print(f"  image transfer       : {'copy' if args.copy_images else 'symlink'}")

    modalities = detect_modalities(nnunet_dir / "dataset.json")
    if modalities:
        print(f"  modalities (auto)    : {modalities}")
    else:
        print("  modalities (auto)    : not found -> using placeholders")
    print()

    # ---- labelsTr (required) ----
    seg_tr_dir = nnunet_dir / "labelsTr"
    if not seg_tr_dir.is_dir():
        raise FileNotFoundError(f"labelsTr not found: {seg_tr_dir}")
    totals_tr = convert_split(seg_tr_dir,
                              raw_splitted / "labelsTr",
                              "labelsTr")

    # ---- labelsTs (optional) ----
    seg_ts_dir = nnunet_dir / "labelsTs"
    has_test_labels = (seg_ts_dir.is_dir()
                       and any(seg_ts_dir.glob("*.nii.gz")))
    totals_ts: Optional[Dict[str, int]] = None
    if has_test_labels:
        totals_ts = convert_split(seg_ts_dir,
                                  raw_splitted / "labelsTs",
                                  "labelsTs")

    # ---- imagesTr (required) ----
    img_tr_src = nnunet_dir / "imagesTr"
    if not img_tr_src.is_dir():
        raise FileNotFoundError(f"imagesTr not found: {img_tr_src}")
    n_img_tr = link_or_copy_dir(img_tr_src, raw_splitted / "imagesTr",
                                copy=args.copy_images)
    print(f"\n  imagesTr: {n_img_tr} files "
          f"{'copied' if args.copy_images else 'symlinked'}")

    # ---- imagesTs (optional) ----
    img_ts_src = nnunet_dir / "imagesTs"
    if img_ts_src.is_dir() and any(img_ts_src.glob("*.nii.gz")):
        n_img_ts = link_or_copy_dir(img_ts_src, raw_splitted / "imagesTs",
                                    copy=args.copy_images)
        print(f"  imagesTs: {n_img_ts} files "
              f"{'copied' if args.copy_images else 'symlinked'}")

    # ---- dataset.json ----
    ds_json_path = write_dataset_json(task_dir, args.task_name, modalities,
                                       has_test_labels=has_test_labels)
    print(f"  dataset.json -> {ds_json_path}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    _print_totals("labelsTr", totals_tr)
    if totals_ts is not None:
        _print_totals("labelsTs", totals_ts)
    else:
        print("\n  [labelsTs] (not present, skipped)")


def _print_totals(split: str, t: Dict[str, int]) -> None:
    def pct(n: int, d: int) -> str:
        return f"{n/d*100:.1f}%" if d > 0 else "n/a"
    print(f"\n  [{split}]")
    print(f"    cases written            : {t['n_cases']}")
    print(f"    cases failed             : {t['n_failed']}")
    print(f"    cases with no ET lesion  : {t['n_empty']}")
    print(f"    cases with small (cls 0) : {t['n_with_small']}")
    print(f"    cases with large (cls 1) : {t['n_with_large']}")
    print(f"    cases with both classes  : {t['n_with_both']}")
    print(f"    total ET lesions         : {t['n_lesions']}")
    print(f"      class 0 (small <27)    : {t['n_small']:>6} "
          f"({pct(t['n_small'], t['n_lesions'])})")
    print(f"      class 1 (large >=27)   : {t['n_large']:>6} "
          f"({pct(t['n_large'], t['n_lesions'])})")


if __name__ == "__main__":
    main()