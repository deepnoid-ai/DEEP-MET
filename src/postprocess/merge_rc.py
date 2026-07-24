"""
Merge two models' segmentation outputs, B-PRIORITY variant.

Rule (per case, matched by filename):
  * From model B : keep EVERYTHING as-is (all classes, including B's own small RC). Nothing removed.
  * From model A : take ONLY its small RC components (volume <= --max-vol mm^3).
  * Merge        : result = B (unchanged), then add A's small RC ONLY where B is background (0).
                   B has full priority: any voxel B already labels (RC or any other class) is
                   preserved and NEVER overwritten; A only fills B's empty background.

Difference vs merge_rc.py (A-priority): there B's small RC is removed and A's small RC overwrites
whatever B had. Here B is kept entirely and A only supplements B's background.

Voxel volume is derived from each image's affine (BraTS 1x1x1 mm -> 27 mm^3 == 27 voxels).

Usage:
    python merge_rc_b_priority.py --a-dir <A_pred_dir> --b-dir <B_pred_dir> --output-dir <out_dir>
    python merge_rc_b_priority.py --a-dir A --b-dir B --output-dir OUT --max-vol 27 --rc-label 4
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume in mm^3 from an affine matrix."""
    voxel_dims = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(np.prod(voxel_dims))


def small_rc_labeled(seg: np.ndarray, rc_label: int, voxel_vol: float,
                     max_vol_mm3: float) -> Tuple[np.ndarray, list]:
    """Label RC (rc_label) connected components and return (labeled_array, small_comp_ids),
    where small_comp_ids are the component labels whose volume <= max_vol_mm3.
    labeled_array is None if there is no RC."""
    binary = (seg == rc_label)
    if not binary.any():
        return None, []
    labeled, n = ndimage.label(binary)               # 6-connectivity (matches postprocess.py)
    if n == 0:
        return None, []
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))  # voxel counts per component
    small_ids = [i for i, sz in enumerate(sizes, start=1) if sz * voxel_vol <= max_vol_mm3]
    return labeled, small_ids


def merge_case(a_path: Path, b_path: Path, out_path: Path,
               rc_label: int, max_vol_mm3: float, overlap_mode: str = "voxel") -> None:
    a_img = nib.load(str(a_path))
    b_img = nib.load(str(b_path))
    a_data = np.asarray(a_img.dataobj).astype(np.int32)
    b_data = np.asarray(b_img.dataobj).astype(np.int32)

    if a_data.shape != b_data.shape:
        raise ValueError(f"Shape mismatch for {a_path.name}: A{a_data.shape} vs B{b_data.shape}")

    vvol_a = voxel_volume_mm3(a_img.affine)

    # B: kept entirely as-is (all classes, including B's own small RC). Nothing removed.
    result = b_data.copy()
    b_object = (b_data != 0)

    labeled_a, small_ids = small_rc_labeled(a_data, rc_label, vvol_a, max_vol_mm3)
    n_a = len(small_ids)
    n_added_vox = 0
    n_added_comp = 0
    n_dropped_comp = 0    # component mode: whole components discarded for touching a B label

    if overlap_mode == "component":
        # all-or-nothing: an A small-RC component is added ONLY if it does NOT touch any B label.
        for cid in small_ids:
            comp = (labeled_a == cid)
            if np.any(comp & b_object):
                n_dropped_comp += 1          # overlaps a B label -> drop the WHOLE component
            else:
                result[comp] = rc_label       # fully inside B background -> add entire component
                n_added_vox += int(comp.sum())
                n_added_comp += 1
        detail = (f"added {n_added_comp}/{n_a} components ({n_added_vox} vox), "
                  f"dropped {n_dropped_comp} components touching B labels")
    else:  # "voxel": add A small RC only on B-background voxels (partial components get clipped)
        if labeled_a is not None and small_ids:
            a_small = np.isin(labeled_a, small_ids)
        else:
            a_small = np.zeros_like(b_data, dtype=bool)
        add_mask = a_small & (~b_object)
        n_added_vox = int(add_mask.sum())
        n_blocked_vox = int(a_small.sum()) - n_added_vox
        result[add_mask] = rc_label
        detail = (f"added {n_added_vox} voxels into B background, "
                  f"{n_blocked_vox} voxels blocked by B labels")

    out_img = nib.Nifti1Image(result.astype(np.int32), b_img.affine, b_img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    print(f"{out_path.name}: B kept as-is; A had {n_a} small RC [{overlap_mode}] -> {detail} "
          f"(<= {max_vol_mm3:g} mm^3)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge two models' outputs, size-based RC hand-off (B-priority).")
    p.add_argument("--a-dir", required=True, type=Path, help="Dir with model A predictions (.nii.gz).")
    p.add_argument("--b-dir", required=True, type=Path, help="Dir with model B predictions (.nii.gz).")
    p.add_argument("--output-dir", required=True, type=Path, help="Output dir for merged predictions.")
    p.add_argument("--rc-label", type=int, default=4, help="RC class label (default: 4).")
    p.add_argument("--max-vol", type=float, default=99999.0,
                   help="Small-RC volume threshold in mm^3 (inclusive, default: 27).")
    p.add_argument("--overlap-mode", choices=["voxel", "component"], default="component",
                   help="How A's small RC is added over B background. 'voxel' (default): add only "
                        "B-background voxels (a component overlapping a B label is clipped). "
                        "'component': all-or-nothing -- if a component touches ANY B label, the "
                        "WHOLE component is dropped; only components fully inside B background are added.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b_files = sorted(f for f in args.b_dir.glob("*.nii.gz") if "probs" not in f.name)
    n = 0
    for b_path in b_files:
        a_path = args.a_dir / b_path.name
        if not a_path.is_file():
            print(f"[skip] no matching A file for {b_path.name}")
            continue
        merge_case(a_path, b_path, args.output_dir / b_path.name, args.rc_label, args.max_vol,
                   overlap_mode=args.overlap_mode)
        n += 1
    print(f"Done. Merged {n} cases -> {args.output_dir}")


if __name__ == "__main__":
    main()
