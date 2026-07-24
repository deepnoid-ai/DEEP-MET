"""
Split a BraTS-MET dataset (nnU-Net raw layout) into two coordinate spaces:
SRI24 (registered, fixed 240x240x155 grid) and Native (everything else).

Each case's image shape is inspected: images matching SRI24_SHAPE
(240, 240, 155) are tagged "SRI24", all others "Native". Every file
(imagesTr/imagesTs multi-modality images, labelsTr/labelsTs segmentations,
and metadata such as dataset.json) is then routed into
<output_root>/<SRI24|Native>/<task_name>/<split>/ via symlink, copy, or move.

Input  (existing, nnU-Net raw layout):
    <dataset_dir>/                       (e.g. Dataset001_BraTSMET)
        imagesTr/<case>_0000.nii.gz      (one file per modality: _0000.._0003)
        imagesTs/<case>_0000.nii.gz
        labelsTr/<case>.nii.gz
        labelsTs/<case>.nii.gz
        dataset.json

Output (created by this script):
    <output_root>/
        SRI24/<task_name>/
            imagesTr/  imagesTs/  labelsTr/  labelsTs/  dataset.json
        Native/<task_name>/
            imagesTr/  imagesTs/  labelsTr/  labelsTs/  dataset.json

Classification rule:
    image shape == (240, 240, 155)  ->  "SRI24"   (registered atlas space)
    image shape != (240, 240, 155)  ->  "Native"  (original acquisition grid)
    (a case that fails to load is conservatively tagged "Native")

Transfer modes:
    symlink (default) -> saves disk, points back to the originals
    copy              -> safe, duplicates the files
    move              -> empties the source (metadata is still copied, not moved)

Usage:
    python split_sri_native.py \
        --dataset_dir /ai-data1/workspace/BraTS-Met/Dataset/Dataset001_BraTSMET \
        --output_root /ai-data1/workspace/BraTS-Met/Dataset \
        --mode copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SRI24_SHAPE: Tuple[int, int, int] = (240, 240, 155)
IMAGE_SUFFIX = "_0000.nii.gz"
SPLITS_WITH_MODALITY = ("imagesTr", "imagesTs")
SPLITS_LABEL_ONLY = ("labelsTr", "labelsTs")


def get_case_id_from_image(filename: str) -> str:
    """BraTS-MET-00001-000_0000.nii.gz -> BraTS-MET-00001-000"""
    return filename.replace("_0000.nii.gz", "").replace("_0001.nii.gz", "") \
                   .replace("_0002.nii.gz", "").replace("_0003.nii.gz", "")


def get_case_id_from_label(filename: str) -> str:
    """BraTS-MET-00001-000.nii.gz -> BraTS-MET-00001-000"""
    return filename.replace(".nii.gz", "")


def classify_cases(dataset_dir: Path) -> Dict[str, str]:
    """Inspect every modality-0 image and tag its case as SRI24 or Native
    based on whether its 3D shape equals SRI24_SHAPE (240x240x155)."""
    classification: Dict[str, str] = {}
    shape_counts: Dict[Tuple, int] = defaultdict(int)

    for split in SPLITS_WITH_MODALITY:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        files = sorted(split_dir.glob(f"*{IMAGE_SUFFIX}"))
        for f in tqdm(files, desc=f"Inspecting {split}"):
            case_id = f.name.replace(IMAGE_SUFFIX, "")
            try:
                shape = tuple(nib.load(str(f)).shape[:3])
            except Exception as e:
                print(f"  WARNING: failed to read {f.name}: {e}")
                classification[case_id] = "Native"  # fail-safe default
                continue
            shape_counts[shape] += 1
            classification[case_id] = "SRI24" if shape == SRI24_SHAPE else "Native"

    print("\n=== Shape distribution ===")
    for shape, n in sorted(shape_counts.items(), key=lambda x: -x[1]):
        tag = "  <- SRI24" if shape == SRI24_SHAPE else ""
        print(f"  {shape}: {n}{tag}")

    return classification


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    """Place src at dst using the requested mode: 'symlink' | 'copy' | 'move'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode: {mode}")


def distribute_files(dataset_dir: Path,
                     classification: Dict[str, str],
                     output_root: Path,
                     task_name: str,
                     mode: str) -> Dict[str, Dict[str, int]]:
    """Route every image and label file into its case's SRI24/Native folder,
    preserving the imagesTr/imagesTs/labelsTr/labelsTs split structure."""
    counts: Dict[str, Dict[str, int]] = {
        "SRI24": defaultdict(int),
        "Native": defaultdict(int),
    }

    # imagesTr, imagesTs - files carry a _000X modality suffix
    for split in SPLITS_WITH_MODALITY:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        all_files = sorted(split_dir.glob("*.nii.gz"))
        for f in tqdm(all_files, desc=f"Distributing {split}"):
            stem = f.name.replace(".nii.gz", "")
            # strip the trailing _000X modality tag to recover the case id
            case_id = stem[:-5] if stem[-5:].startswith("_0") else stem
            target_space = classification.get(case_id)
            if target_space is None:
                print(f"  WARNING: case {case_id} not classified, skipping {f.name}")
                continue
            dst = output_root / target_space / task_name / split / f.name
            transfer_file(f, dst, mode)
            counts[target_space][split] += 1

    # labelsTr, labelsTs - one segmentation file per case, no modality suffix
    for split in SPLITS_LABEL_ONLY:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        all_files = sorted(split_dir.glob("*.nii.gz"))
        for f in tqdm(all_files, desc=f"Distributing {split}"):
            case_id = f.name.replace(".nii.gz", "")
            target_space = classification.get(case_id)
            if target_space is None:
                print(f"  WARNING: case {case_id} not classified, skipping {f.name}")
                continue
            dst = output_root / target_space / task_name / split / f.name
            transfer_file(f, dst, mode)
            counts[target_space][split] += 1

    return counts


def copy_metadata_files(dataset_dir: Path, output_root: Path,
                        task_name: str, mode: str) -> None:
    """Duplicate dataset.json and related metadata into BOTH space folders.
    Metadata is always copied (never moved) so the source stays intact."""
    meta_patterns = ["dataset.json", "dataset_*.json",
                     "split_info.json", "split_info.csv"]
    for space in ("SRI24", "Native"):
        target_dir = output_root / space / task_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for pat in meta_patterns:
            for src in dataset_dir.glob(pat):
                if src.is_file():
                    dst = target_dir / src.name
                    if mode == "move":
                        shutil.copy2(src, dst)
                    else:
                        transfer_file(src, dst, "copy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split BraTS-MET into SRI24 (240x240x155) and Native space.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Source dataset folder. e.g. Dataset/Dataset001_BraTSMET")
    parser.add_argument("--output_root", required=True,
                        help="Output root folder. e.g. Dataset or Dataset_split")
    parser.add_argument("--task_name", default=None,
                        help="Task folder name. Defaults to dataset_dir's basename.")
    parser.add_argument("--mode", choices=["symlink", "copy", "move"],
                        default="symlink",
                        help="symlink (default, saves disk), copy (safe), "
                             "move (empties source).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    task_name = args.task_name or dataset_dir.name

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    print("=" * 72)
    print("BraTS-MET SRI24/Native split")
    print("=" * 72)
    print(f"  Input dataset  : {dataset_dir}")
    print(f"  Output root    : {output_root}")
    print(f"  Task name      : {task_name}")
    print(f"  SRI24 shape    : {SRI24_SHAPE}")
    print(f"  Mode           : {args.mode}")
    print()

    # 1. Classify each case by image shape
    print("=== Step 1: classify cases ===\n")
    classification = classify_cases(dataset_dir)
    n_sri24 = sum(1 for v in classification.values() if v == "SRI24")
    n_native = sum(1 for v in classification.values() if v == "Native")
    print(f"\n  SRI24 cases  : {n_sri24}")
    print(f"  Native cases : {n_native}")
    print(f"  Total        : {len(classification)}")

    # 2. Distribute image/label files
    print("\n=== Step 2: distribute files ===\n")
    counts = distribute_files(dataset_dir, classification,
                              output_root, task_name, args.mode)

    # 3. Copy metadata files into both spaces
    print("\n=== Step 3: copy metadata files ===\n")
    copy_metadata_files(dataset_dir, output_root, task_name, args.mode)
    print("  done.")

    # 4. Summary
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for space in ("SRI24", "Native"):
        print(f"\n  [{space}] -> {output_root / space / task_name}")
        for split, n in counts[space].items():
            print(f"    {split}: {n} files")


if __name__ == "__main__":
    main()