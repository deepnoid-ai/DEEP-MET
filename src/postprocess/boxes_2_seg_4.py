"""
nnUNet small-ET FP cleanup using nnDetection BOUNDING BOXES (SUBTRACTIVE, 2-class).

2-class 버전: nnDetection pred_label 이 small(0) / large(1) 로 나오는 모델용.
single-class 버전과 핵심 동작은 같되, 작은 ET CC 를 confirm 할 때 SMALL 박스
(pred_label == --small_class) 만 사용한다.

이 스크립트는 nnDetection 의 segmentation(_seg.pkl)을 쓰지 않는다.
오직 *_boxes.pkl 의 bbox 만 "이 자리에 진짜 (작은) ET 병변이 있다"는 confirm
신호로 사용하여, nnUNet 결과에서 작은 ET false-positive 만 제거한다 (additive 없음).

핵심 규칙 (nnUNet ET CC 별):
    - 부피 >= --et_thresh (큰 것)  → 무조건 KEEP (nnUNet 신뢰; 안 건드림)
    - 부피 <  --et_thresh (작은 것) → SMALL 박스로 confirm 되면 KEEP, 아니면 DELETE
  large 박스(pred_label == large)는 이 FP reduction 에서 쓰지 않는다
  (작은 CC 를 large 박스로 살릴 이유가 없으므로). large CC 는 어차피 무조건 KEEP.

추가로, 다른 merge 스크립트와 동일한 PER-LABEL small-lesion cleanup 을 먼저
적용할 수 있다 (--thresh-netc/-snfh/-et/-rc). 이는 bbox 와 무관하게 각 라벨의
아주 작은 CC 를 먼저 제거한다 (vol < thresh, strictly-less-than).

동작 (per case):
  0. (Optional) PER-LABEL cleanup: NETC/SNFH/ET/RC 각각 자기 binary mask 에서
     26-CC 떠서 vol < 그 라벨 threshold 인 CC 삭제 (bbox 무관). 0 = off.
  1. nnUNet seg 로드 (affine/header 보존).
  2. *_boxes.pkl 로드 → SMALL 박스(pred_label == --small_class)만, --score_thresh 로 필터.
  3. 살아남은 small box 들을 nnUNet voxel grid 에 래스터화하여 bbox_mask(bool) 생성.
     (--bbox_pad 로 각 박스를 voxel 단위로 팽창 가능; 기본 0)
  4. ET binary mask (seg == 3) 를 26-connectivity 로 CC 분해.
  5. 각 ET CC 에 대해:
       - 부피 >= --et_thresh           → KEEP (큰 병변은 nnUNet 신뢰; 안 건드림)
       - 부피 <  --et_thresh (작은 것) → small bbox confirm 여부 확인
            confirm(--confirm_mode)  → KEEP
            unconfirmed              → DELETE (그 CC 의 ET voxel 들을 0 으로)
  6. 결과 저장.

⚠️ PER-LABEL cleanup 과 bbox FP cleanup 의 관계:
  순서는 (0) per-label cleanup -> (4~5) bbox FP cleanup 이다. 둘 다 ET 를 건드린다:
    - per-label cleanup 은 vol < --thresh-et 인 ET CC 를 bbox 무관하게 삭제.
    - bbox FP cleanup 은 vol < --et_thresh 인 ET CC 중 small bbox unconfirmed 만 삭제.
  보통 --thresh-et(아주 작은 noise 제거, 예: 5) <= --et_thresh(small 경계, 예: 27)
  로 두면, 먼저 noise 를 치우고 남은 small ET 중 미confirm 을 FP cleanup 이 지운다.

왜 ET 만 (bbox FP cleanup)?
  nnDetection 이 ET-only 로 학습됐으므로 bbox 는 ET 병변에만 존재한다. 따라서
  "bbox 밖 = FP" 신호는 ET 라벨에만 유효하다. NETC/SNFH/RC 는 nnDetection 이
  애초에 박스를 안 치므로 이 필터를 적용하면 전부 삭제돼버린다 → 절대 금지.
  (PER-LABEL cleanup 은 bbox 와 무관한 순수 크기 필터라 NETC/SNFH/RC 에도 안전.)

⚠️ Score threshold 의미가 additive 때와 반대다:
  --score_thresh 를 올리면 confirm 되는 box 가 줄어 → 삭제가 더 공격적이 된다
  (FP 를 더 지우지만 TP 오삭제 위험↑). 낮추면 보수적(덜 삭제).

⚠️ 좌표 정합:
  bbox 는 nnDetection 전처리 voxel grid 기준이다. nnUNet seg 와 동일한 geometry
  (같은 space/shape) 여야 voxel index 가 맞는다. Native task 결과에는 Native nnUNet,
  SRI24 task 결과에는 SRI24 nnUNet 을 짝지어라. box center 가 seg 범위를 크게
  벗어나면 space 불일치 경고를 출력한다.

Coordinate conventions (원본 스크립트와 동일, _boxes_gt.npz 로 검증됨):
  nnDetection 6-tuple "(x1, y1, x2, y2, z1, z2)" (SimpleITK axis):
      box[0], box[2]  -> nibabel axis 2
      box[1], box[3]  -> nibabel axis 1
      box[4], box[5]  -> nibabel axis 0

Usage:

    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task008_Native_full_ET/RetinaUNetV001_D3V001_3d/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/120 \
        --score_thresh 0.3 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5

    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task009_SRI24_full_ET/RetinaUNetV001_D3V001_3d/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/120 \
        --score_thresh 0.3 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task010_Native_full_ET_new_loss/RetinaUNetV100_D3V001_3d_C16/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/121 \
        --score_thresh 0.15 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5

    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task011_SRI24_full_ET_new_loss/RetinaUNetV100_D3V001_3d_C16/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/121 \
        --score_thresh 0.15 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ        
    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task012_Native_full_ET_class_condition/RetinaUNetV101_D3V001_3d/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/122 \
        --score_thresh 0.15 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5

    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task013_SRI24_full_ET_class_condition/RetinaUNetV101_D3V001_3d/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/107 \
        --output_dir ./Combined_results/122 \
        --score_thresh 0.15 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    python final_2_classes_ET_fp_reduction.py \
        --pred_dir   ./Task001_BraTSMET/RetinaUNetV001_D3V001_3d/consolidated/test_predictions \
        --seg_dir    ./nnUNet_results/Dataset001_5mm \
        --output_dir ./Combined_results/BraTSMET \
        --score_thresh 0.15 \
        --et_thresh    27 \
        --small_class  0 \
        --confirm_mode center \
        --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

try:
    import cc3d
    _HAS_CC3D = True
except ImportError:
    _HAS_CC3D = False


# BraTS-MET raw labels
LABELS = {"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}
LABEL_NAME = {v: k for k, v in LABELS.items()}
ET_LABEL = 3  # BraTS-MET enhancing tumor


# ════════════════════════════════════════════════════════════════════
# 26-connectivity connected components (matches eval Panoptica)
# ════════════════════════════════════════════════════════════════════
def cc26(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Returns (label_array, n_components) with 26-connectivity, bg=0."""
    mask = mask.astype(np.uint8, copy=False)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    if _HAS_CC3D:
        lab = cc3d.connected_components(mask, connectivity=26).astype(np.int32)
        return lab, int(lab.max())
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = ndi.label(mask, structure=struct)
    return lab.astype(np.int32), int(n)


# ════════════════════════════════════════════════════════════════════
# nnUNet small-lesion cleanup — PER LABEL (eval-consistent, bbox 무관)
# ════════════════════════════════════════════════════════════════════
def remove_small_lesions_per_label(seg: np.ndarray,
                                   label_thresholds: Dict[int, float],
                                   voxel_vol_mm3: float
                                   ) -> Tuple[int, int, int, Dict[int, int]]:
    """Delete (set to 0) CCs whose physical volume < threshold, processing
    EACH label on its OWN binary mask (seg == label), 26-connectivity.
    MODIFIES seg IN PLACE.

    label_thresholds: {label_value: threshold_mm3}. threshold <= 0 -> skip
    (label not filtered, but its CCs are still counted in n_total).

    부등호는 strictly-less-than (vol < thresh): official 'small' 정의와 일관.
    각 라벨 독립적으로 CC 를 떠서 cross-label merging 이 없다 (official eval 과 일관).

    Returns (n_total, n_removed, voxels_removed, removed_per_label).
    """
    n_total = n_removed = voxels_removed = 0
    removed_per_label: Dict[int, int] = {lv: 0 for lv in label_thresholds}

    for label_val, thresh in label_thresholds.items():
        binary = (seg == label_val)         # THIS label only -> no merging
        lab, n = cc26(binary)
        if n == 0:
            continue
        n_total += n
        if thresh <= 0:
            continue  # not filtered, but counted
        slices = ndi.find_objects(lab)
        for i, sl in enumerate(slices):
            if sl is None:
                continue
            gid = i + 1
            cc_mask = (lab[sl] == gid)
            n_vox = int(cc_mask.sum())
            vol_mm3 = float(n_vox) * voxel_vol_mm3
            if vol_mm3 < thresh:            # strictly-less-than (eval small def)
                seg_chunk = seg[sl]
                seg_chunk[cc_mask] = 0
                n_removed += 1
                voxels_removed += n_vox
                removed_per_label[label_val] += 1
    return n_total, n_removed, voxels_removed, removed_per_label


# ════════════════════════════════════════════════════════════════════
# Rasterize filtered nnDetection boxes into a boolean mask on the seg grid
# ════════════════════════════════════════════════════════════════════
def build_bbox_mask(boxes_kept: np.ndarray,
                    shape: Tuple[int, int, int],
                    pad: int = 0) -> Tuple[np.ndarray, int]:
    """Union of all kept bbox interiors as a bool array of `shape`.

    box layout: [a2_lo, a1_lo, a2_hi, a1_hi, a0_lo, a0_hi]  (see docstring)
    pad: dilate each box by this many voxels on every side (leniency knob).

    Returns (bbox_mask, n_centers_out_of_bounds) where the second value counts
    boxes whose center falls outside `shape` (space-mismatch sanity check).
    """
    bbox_mask = np.zeros(shape, dtype=bool)
    if boxes_kept.shape[0] == 0:
        return bbox_mask, 0

    s0, s1, s2 = shape
    n_oob = 0
    for b in boxes_kept:
        a2_lo, a1_lo, a2_hi, a1_hi, a0_lo, a0_hi = (
            float(b[0]), float(b[1]), float(b[2]),
            float(b[3]), float(b[4]), float(b[5]))
        # ensure lo <= hi on each axis
        a0_lo, a0_hi = sorted((a0_lo, a0_hi))
        a1_lo, a1_hi = sorted((a1_lo, a1_hi))
        a2_lo, a2_hi = sorted((a2_lo, a2_hi))

        # center for the out-of-bounds sanity check
        c0 = 0.5 * (a0_lo + a0_hi)
        c1 = 0.5 * (a1_lo + a1_hi)
        c2 = 0.5 * (a2_lo + a2_hi)
        if not (0 <= c0 < s0 and 0 <= c1 < s1 and 0 <= c2 < s2):
            n_oob += 1

        lo0 = max(0, int(np.floor(a0_lo)) - pad)
        lo1 = max(0, int(np.floor(a1_lo)) - pad)
        lo2 = max(0, int(np.floor(a2_lo)) - pad)
        hi0 = min(s0, int(np.ceil(a0_hi)) + 1 + pad)
        hi1 = min(s1, int(np.ceil(a1_hi)) + 1 + pad)
        hi2 = min(s2, int(np.ceil(a2_hi)) + 1 + pad)
        if lo0 < hi0 and lo1 < hi1 and lo2 < hi2:
            bbox_mask[lo0:hi0, lo1:hi1, lo2:hi2] = True

    return bbox_mask, n_oob


# ════════════════════════════════════════════════════════════════════
# Subtractive cleanup: delete small unconfirmed ET CCs
# ════════════════════════════════════════════════════════════════════
def cleanup_small_et_fp(seg: np.ndarray,
                        bbox_mask: np.ndarray,
                        voxel_vol_mm3: float,
                        et_thresh_mm3: float,
                        confirm_mode: str) -> dict:
    """MODIFIES seg IN PLACE. Returns stats dict.

    For every ET (label 3) connected component:
        vol >= et_thresh_mm3  -> keep (large; trust nnUNet)
        vol <  et_thresh_mm3  -> keep iff confirmed by a SMALL bbox, else delete -> 0
    bbox_mask 는 SMALL 박스만 래스터화한 마스크여야 한다 (호출부에서 보장).
    confirm_mode:
        'center'  : CC centroid voxel lies inside bbox_mask
        'overlap' : any CC voxel lies inside bbox_mask  (more lenient -> deletes less)
    """
    stats = {"n_et_cc": 0, "n_small": 0, "n_large_kept": 0,
             "n_confirmed_kept": 0, "n_deleted": 0, "voxels_deleted": 0}

    et_bin = (seg == ET_LABEL)
    lab, n = cc26(et_bin)
    stats["n_et_cc"] = n
    if n == 0:
        return stats

    slices = ndi.find_objects(lab)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        gid = i + 1
        cc_mask = (lab[sl] == gid)
        n_vox = int(cc_mask.sum())
        vol_mm3 = float(n_vox) * voxel_vol_mm3

        if vol_mm3 >= et_thresh_mm3:        # large -> keep untouched
            stats["n_large_kept"] += 1
            continue

        stats["n_small"] += 1               # small -> subject to small-bbox confirm

        if confirm_mode == "overlap":
            confirmed = bool(bbox_mask[sl][cc_mask].any())
        else:  # 'center'
            local_com = ndi.center_of_mass(cc_mask)
            g0 = int(round(sl[0].start + local_com[0]))
            g1 = int(round(sl[1].start + local_com[1]))
            g2 = int(round(sl[2].start + local_com[2]))
            g0 = min(max(g0, 0), seg.shape[0] - 1)
            g1 = min(max(g1, 0), seg.shape[1] - 1)
            g2 = min(max(g2, 0), seg.shape[2] - 1)
            confirmed = bool(bbox_mask[g0, g1, g2])

        if confirmed:
            stats["n_confirmed_kept"] += 1
        else:
            seg_chunk = seg[sl]
            seg_chunk[cc_mask] = 0          # delete this small ET FP
            stats["n_deleted"] += 1
            stats["voxels_deleted"] += n_vox

    return stats


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, type=Path,
                    help="Folder with nnDetection *_boxes.pkl files")
    ap.add_argument("--seg_dir", required=True, type=Path,
                    help="Folder with nnUNet segmentation *.nii.gz files")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Output folder for cleaned *.nii.gz files")
    # box filtering — 2-class: small 박스만 confirm 에 사용
    ap.add_argument("--small_class", type=int, default=0,
                    help="pred_label value that means SMALL (default 0). 작은 ET CC "
                         "confirm 에는 이 class 박스만 쓴다.")
    ap.add_argument("--score_thresh", type=float, default=0.0,
                    help="Confirm with SMALL boxes whose pred_score >= this. "
                         "HIGHER => more aggressive deletion (fewer confirms). "
                         "(default 0.0)")
    ap.add_argument("--bbox_pad", type=int, default=0,
                    help="Dilate each bbox by this many voxels per side before "
                         "confirming. Larger => more lenient keep, fewer "
                         "deletions (default 0).")
    # PER-LABEL nnUNet cleanup (bbox 무관). 0 = don't filter that label.
    ap.add_argument("--thresh-netc", type=float, default=0.0,
                    help="NETC(1) cleanup threshold mm³. Remove CCs < this. 0 = off.")
    ap.add_argument("--thresh-snfh", type=float, default=0.0,
                    help="SNFH(2) cleanup threshold mm³. Remove CCs < this. 0 = off.")
    ap.add_argument("--thresh-et",   type=float, default=0.0,
                    help="ET(3) cleanup threshold mm³. Remove CCs < this (bbox 무관). "
                         "0 = off. (bbox FP cleanup 전에 적용)")
    ap.add_argument("--thresh-rc",   type=float, default=0.0,
                    help="RC(4) cleanup threshold mm³. Remove CCs < this. 0 = off.")
    # ET bbox FP cleanup
    ap.add_argument("--et_thresh", type=float, default=20.0,
                    help="ET CC volume threshold (mm³). Only CCs with volume < "
                         "this are deletion candidates; >= are always kept. "
                         "Default 20.0 matches the official eval small/large "
                         "boundary (strictly-less-than).")
    ap.add_argument("--confirm_mode", choices=["center", "overlap"],
                    default="center",
                    help="'center': CC centroid must be inside a SMALL bbox to keep. "
                         "'overlap': any CC voxel inside a SMALL bbox keeps it "
                         "(more lenient; deletes fewer). Default 'center'.")
    ap.add_argument("--skip_if_no_boxes", action="store_true",
                    help="If a case has ZERO kept SMALL boxes, skip the bbox ET FP "
                         "cleanup (keep all nnUNet ET) instead of deleting every "
                         "small ET CC. PER-LABEL cleanup still runs. Use this if "
                         "nnDetection sometimes outputs nothing for a valid case.")
    args = ap.parse_args()

    # ─── Per-label cleanup thresholds (mm³) ─────────────────────────
    thr: Dict[int, float] = {LABELS["NETC"]: args.thresh_netc,
                             LABELS["SNFH"]: args.thresh_snfh,
                             LABELS["ET"]:   args.thresh_et,
                             LABELS["RC"]:   args.thresh_rc}
    cleanup_on = any(t > 0 for t in thr.values())

    print("=" * 72)
    print("nnUNet small-ET FP cleanup via nnDetection SMALL bboxes (SUBTRACTIVE, 2-class)")
    print("=" * 72)
    print(f"[info] cc3d available : {_HAS_CC3D}")
    if cleanup_on:
        print(f"[info] PER-LABEL cleanup ON (bbox 무관, FP cleanup 전 적용): "
              f"remove CC with volume < threshold")
        for lv in (LABELS["NETC"], LABELS["ET"], LABELS["RC"], LABELS["SNFH"]):
            state = "skip (no filter)" if thr[lv] <= 0 else f"remove < {thr[lv]:g} mm³"
            print(f"           {LABEL_NAME[lv]:<4} (label {lv}): {state}")
    else:
        print(f"[info] PER-LABEL cleanup OFF (all per-label thresholds = 0)")
    print(f"[info] confirm box    : SMALL only (pred_label == {args.small_class}) AND "
          f"pred_score >= {args.score_thresh}")
    print(f"[info] bbox_pad       : {args.bbox_pad} voxel(s)")
    print(f"[info] ET FP threshold: delete ET CC with volume < {args.et_thresh} mm³ "
          f"if small-bbox-unconfirmed (strictly-less-than). vol >= {args.et_thresh} -> "
          f"always KEEP")
    print(f"[info] confirm_mode   : {args.confirm_mode}")
    print(f"[info] skip_if_no_box : {args.skip_if_no_boxes}")
    print(f"[info] NOTE: bbox FP cleanup 은 ET(label {ET_LABEL}) 의 작은 CC 만 건드림. "
          f"large 박스(non-small)는 쓰지 않음. PER-LABEL cleanup 은 크기 필터(bbox 무관).\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(args.pred_dir.glob("*_boxes.pkl"))
    print(f"[info] {len(pred_files)} prediction files found\n")

    t_total_start = time.perf_counter()
    per_case_times = []

    n_done = n_missing = 0
    n_cases_no_box = n_cases_skipped = 0
    tot_et_cc = tot_small = tot_large_kept = 0
    tot_confirmed = tot_deleted = tot_vox_deleted = 0
    tot_small_boxes_kept = 0
    # per-label cleanup totals
    tot_ccs_seen = tot_ccs_removed = tot_cleanup_vox = 0
    tot_removed_per_label: Dict[int, int] = {lv: 0 for lv in thr}

    for pkl in pred_files:
        t_case_start = time.perf_counter()
        cid = pkl.stem.replace("_boxes", "")
        seg_path = args.seg_dir / f"{cid}.nii.gz"

        if not seg_path.exists():
            print(f"  [skip] {cid}: missing nnUNet seg")
            n_missing += 1
            continue

        # Load nnUNet seg (preserve native dtype + header)
        seg_nii = nib.load(str(seg_path))
        seg = np.asanyarray(seg_nii.dataobj).copy()
        vol_shape = seg.shape
        spacing = tuple(float(z) for z in seg_nii.header.get_zooms()[:3])
        voxel_vol = float(np.prod(spacing))

        # 0. PER-LABEL cleanup (bbox 무관, in place). bbox FP cleanup 전에 먼저.
        n_ccs_total = n_ccs_removed = n_voxels_removed = 0
        if cleanup_on:
            n_ccs_total, n_ccs_removed, n_voxels_removed, removed_pl = \
                remove_small_lesions_per_label(seg, thr, voxel_vol)
            tot_ccs_seen += n_ccs_total
            tot_ccs_removed += n_ccs_removed
            tot_cleanup_vox += n_voxels_removed
            for lv, c in removed_pl.items():
                tot_removed_per_label[lv] += c

        # Load + filter boxes -> SMALL 박스만 (pred_label == small_class)
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        boxes = np.asarray(d["pred_boxes"], dtype=np.float32)
        scores = np.asarray(d["pred_scores"], dtype=np.float32)
        labels = np.asarray(d["pred_labels"]).astype(np.int64)
        if boxes.size == 0:
            boxes = np.zeros((0, 6), dtype=np.float32)
        elif boxes.ndim == 1 and boxes.shape[0] == 6:
            boxes = boxes.reshape(1, 6)
        if boxes.shape[0] > 0:
            sel = (labels == args.small_class) & (scores >= args.score_thresh)
            small_boxes = boxes[sel]
        else:
            small_boxes = boxes
        tot_small_boxes_kept += int(small_boxes.shape[0])

        if small_boxes.shape[0] == 0:
            n_cases_no_box += 1

        # Optionally skip bbox FP cleanup entirely when no small boxes
        # (PER-LABEL cleanup 은 이미 위에서 적용됨)
        if small_boxes.shape[0] == 0 and args.skip_if_no_boxes:
            out_nii = nib.Nifti1Image(seg, affine=seg_nii.affine,
                                      header=seg_nii.header)
            nib.save(out_nii, str(args.output_dir / f"{cid}.nii.gz"))
            n_done += 1
            n_cases_skipped += 1
            t_case = time.perf_counter() - t_case_start
            per_case_times.append(t_case)
            cleanup_str = (f"perlabel_removed={n_ccs_removed}/{n_ccs_total}  "
                           if cleanup_on else "")
            print(f"  [{cid}] {cleanup_str}no small boxes -> skip bbox FP cleanup "
                  f"(kept all ET)  [t={t_case:.2f}s]")
            continue

        # Build bbox mask on the seg grid (SMALL 박스만)
        bbox_mask, n_oob = build_bbox_mask(small_boxes, vol_shape, args.bbox_pad)
        if small_boxes.shape[0] > 0 and n_oob > 0.5 * small_boxes.shape[0]:
            print(f"  [WARN] {cid}: {n_oob}/{small_boxes.shape[0]} small-box centers "
                  f"out of seg bounds -> possible space/shape mismatch!")

        # Subtractive ET FP cleanup (small bbox 기반)
        st = cleanup_small_et_fp(seg, bbox_mask, voxel_vol,
                                 args.et_thresh, args.confirm_mode)

        # Save
        out_nii = nib.Nifti1Image(seg, affine=seg_nii.affine,
                                  header=seg_nii.header)
        nib.save(out_nii, str(args.output_dir / f"{cid}.nii.gz"))

        n_done += 1
        tot_et_cc += st["n_et_cc"]
        tot_small += st["n_small"]
        tot_large_kept += st["n_large_kept"]
        tot_confirmed += st["n_confirmed_kept"]
        tot_deleted += st["n_deleted"]
        tot_vox_deleted += st["voxels_deleted"]

        t_case = time.perf_counter() - t_case_start
        per_case_times.append(t_case)
        cleanup_str = (f"perlabel_removed={n_ccs_removed}/{n_ccs_total}"
                       f"({n_voxels_removed}vox)  " if cleanup_on else "")
        print(f"  [{cid}] {cleanup_str}"
              f"small_boxes={small_boxes.shape[0]}/{boxes.shape[0]}  "
              f"ET_cc={st['n_et_cc']} (large_kept={st['n_large_kept']}, "
              f"small={st['n_small']})  "
              f"small[confirmed_kept={st['n_confirmed_kept']}, "
              f"deleted={st['n_deleted']}]  "
              f"voxels_deleted={st['voxels_deleted']}  [t={t_case:.2f}s]")

    print(f"\n[done] wrote {n_done} files to {args.output_dir}")
    if cleanup_on:
        print(f"  PER-LABEL cleanup: removed {tot_ccs_removed}/{tot_ccs_seen} CCs "
              f"({tot_cleanup_vox} voxels)")
        for lv in (LABELS["NETC"], LABELS["ET"], LABELS["RC"], LABELS["SNFH"]):
            if thr[lv] > 0:
                print(f"      {LABEL_NAME[lv]:<4} (< {thr[lv]:g} mm³): "
                      f"removed {tot_removed_per_label[lv]}")
    print(f"  total SMALL boxes (class={args.small_class}, after score) : {tot_small_boxes_kept}")
    print(f"  total ET CCs seen (after per-label)  : {tot_et_cc}")
    print(f"     large ET CCs kept (>= {args.et_thresh:g} mm³)   : {tot_large_kept}")
    print(f"     small ET CCs (< {args.et_thresh:g} mm³)         : {tot_small}")
    print(f"        confirmed -> kept                 : {tot_confirmed}")
    print(f"        unconfirmed -> DELETED            : {tot_deleted} "
          f"({tot_vox_deleted} voxels)")
    print(f"  cases with zero kept SMALL boxes     : {n_cases_no_box}"
          + (f" (bbox FP cleanup skipped on {n_cases_skipped})"
             if args.skip_if_no_boxes else ""))
    if n_missing > 0:
        print(f"  missing nnUNet seg (skipped)         : {n_missing}")

    t_total = time.perf_counter() - t_total_start
    h, rem = divmod(t_total, 3600)
    m, s = divmod(rem, 60)
    print(f"\n[timing]")
    print(f"  elapsed : {int(h):02d}:{int(m):02d}:{s:05.2f}  ({t_total:.2f} s)")
    if per_case_times:
        arr = np.asarray(per_case_times)
        print(f"  per-case (n={len(arr)}) : mean={arr.mean():.2f}s  "
              f"median={np.median(arr):.2f}s  min={arr.min():.2f}s  "
              f"max={arr.max():.2f}s")


if __name__ == "__main__":
    main()