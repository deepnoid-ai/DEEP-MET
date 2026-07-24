"""
Build an RC-only BINARY dataset from an existing multi-modality dataset.

Only cases whose label contains class 4 (RC present) are copied. For each such case:
  * imagesTr : copy ALL image channels (_0000, _0001, _0002, _0003, ...) unchanged.
  * labelsTr : remap the integer label map ->
        value 4 (RC)               -> 1
        values 1,2,3 (other tumor) -> 0
        value 0 (background)       -> 0
    i.e. a binary target {0: background/other, 1: RC}.

Images are copied with shutil.copy2 (follows symlinks -> real files at destination).

Usage:
    python make_rc_binary_dataset.py --src-dir <SRC_dataset> --dst-dir <DST_dataset>
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

FILE_ENDING = ".nii.gz"
CHANNEL_GLOB = "_[0-9][0-9][0-9][0-9]"   # nnU-Net 4-digit channel suffix


def remap_label(data: np.ndarray) -> np.ndarray:
    """4 -> 1 (RC),  {1,2,3} -> 0,  0 -> 0 (background)."""
    out = np.zeros_like(data, dtype=np.uint8)
    out[data == 4] = 1
    return out


def process_case(name: str, src_dir: Path, dst_dir: Path) -> bool:
    src_img, src_lab = src_dir / "imagesTr", src_dir / "labelsTr"
    dst_img, dst_lab = dst_dir / "imagesTr", dst_dir / "labelsTr"

    # load label; only keep cases that actually contain RC (class 4)
    lab_img = nib.load(str(src_lab / f"{name}{FILE_ENDING}"))
    data = np.rint(np.asarray(lab_img.dataobj)).astype(np.int32)
    if not np.any(data == 4):
        return False  # no RC -> skip

    # copy ALL image channels for this case
    channels = sorted(src_img.glob(f"{name}{CHANNEL_GLOB}{FILE_ENDING}"))
    if not channels:
        print(f"[skip] no image channels found for {name}")
        return False
    for ch_f in channels:
        shutil.copy2(ch_f, dst_img / ch_f.name)

    # remap + save label
    new = remap_label(data)
    out_img = nib.Nifti1Image(new, lab_img.affine, lab_img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(dst_lab / f"{name}{FILE_ENDING}"))
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="RC-only binary dataset (4->1, 1/2/3->0).")
    p.add_argument("--src-dir", required=True, type=Path,
                   help="Source dataset dir (contains imagesTr/ and labelsTr/).")
    p.add_argument("--dst-dir", required=True, type=Path,
                   help="Destination dataset dir (imagesTr/ and labelsTr/ will be created).")
    args = p.parse_args()

    (args.dst_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (args.dst_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    label_files = sorted((args.src_dir / "labelsTr").glob(f"*{FILE_ENDING}"))
    n_ok = 0
    for lab_f in label_files:
        name = lab_f.name[:-len(FILE_ENDING)]
        if process_case(name, args.src_dir, args.dst_dir):
            n_ok += 1
    print(f"Done. Wrote {n_ok}/{len(label_files)} RC-containing cases -> {args.dst_dir} "
          f"(all channels copied, labels remapped 4->1, 1/2/3->0)")


if __name__ == "__main__":
    main()
