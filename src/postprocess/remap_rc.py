"""
Remap an RC-specialist model's predicted masks to the challenge label (RC = 4).

Two model label schemes are supported via --mode:
  * mode 3-class : model trained as {0: background, 1: non-RC, 2: RC}
             -> 1 is dropped (->0), 2 -> 4.   (result: {0: background, 4: RC})
  * mode binary : model trained as {0: background, 1: RC}
             -> 1 -> 4.                        (result: {0: background, 4: RC})

Everything else becomes background. Output keeps the same affine/header as the input.

Usage:
    python remap_rc_pred_to_challenge.py --input-dir <pred_dir> --output-dir <out_dir> --mode 3-class
    python remap_rc_pred_to_challenge.py --input-dir <pred_dir> --output-dir <out_dir> --mode binary
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

FILE_ENDING = ".nii.gz"


def remap(data: np.ndarray, mode: str, rc_label: int) -> np.ndarray:
    """mode 'a': {1->0, 2->rc_label};  mode 'b': {1->rc_label}. All else -> 0."""
    out = np.zeros_like(data, dtype=np.uint8)
    if mode == "3-class":
        out[data == 2] = rc_label      # RC (label 2 in A) -> challenge RC; label 1 (non-RC) dropped
    else:  # mode "b"
        out[data == 1] = rc_label      # RC (label 1 in B) -> challenge RC
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Remap RC-specialist predictions to challenge label (RC=4).")
    p.add_argument("--input-dir", required=True, type=Path, help="Dir with predicted .nii.gz masks.")
    p.add_argument("--output-dir", required=True, type=Path, help="Output dir for remapped masks.")
    p.add_argument("--mode", required=True, choices=["3-class", "binary"],
                   help="3-class: {0:bg,1:non-rc,2:rc} -> 1 dropped, 2->4.  binary: {0:bg,1:rc} -> 1->4.")
    p.add_argument("--rc-label", type=int, default=4, help="Challenge RC label (default: 4).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in args.input_dir.glob(f"*{FILE_ENDING}") if "probs" not in f.name)
    n = 0
    for f in files:
        img = nib.load(str(f))
        data = np.rint(np.asarray(img.dataobj)).astype(np.int32)
        new = remap(data, args.mode, args.rc_label)
        out_img = nib.Nifti1Image(new, img.affine, img.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(args.output_dir / f.name))
        n += 1
        print(f"{f.name}: mode={args.mode} -> RC({args.rc_label}) voxels = {int((new == args.rc_label).sum())}")
    print(f"Done. Remapped {n} files -> {args.output_dir} (mode {args.mode}, RC label {args.rc_label})")


if __name__ == "__main__":
    main()
