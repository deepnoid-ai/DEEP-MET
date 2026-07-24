"""
prepare_input.py  (BraTS-MET 2026 submission)

/input 의 각 케이스 폴더를 읽어서:
  1. t1c 의 size+spacing 을 보고 Task002_Native vs Task003_SRI24 결정
       - (240, 240, 155) AND spacing ~ (1,1,1) mm  -> Task003_SRI24
       - 그 외                                       -> Task002_Native
  2. 4채널을 nnUNet 입력 형식으로 복사:  <nnunet_in>/{case}_0000..0003.nii.gz
  3. 4채널을 nndet 입력 형식으로 복사:
       $det_data/<task>/raw_splitted/imagesTs/{case}_0000..0003.nii.gz
  4. 케이스->task 매핑을 JSON 으로 기록 (run.sh 가 읽어서 어느 Task 로
     nndet_predict 할지 결정)

BraTS 입력 채널 순서 (dataset.json 과 일치해야 함):
    0000 = t1c, 0001 = t1n, 0002 = t2f, 0003 = t2w

/input 은 read-only 이므로 절대 쓰지 않는다. 모든 출력은 /tmp 또는 $det_data 로.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

# BraTS 파일 suffix -> nnUNet 채널 인덱스
CHANNEL_SUFFIX = {
    "t1c": "0000",
    "t1n": "0001",
    "t2f": "0002",
    "t2w": "0003",
}

SRI24_SHAPE = (240, 240, 155)
SRI24_SPACING_TOL = 0.05  # mm 허용 오차


def is_sri24(t1c_path: Path) -> bool:
    """size==(240,240,155) and spacing~(1,1,1) 이면 SRI24."""
    img = nib.load(str(t1c_path))
    shape = tuple(int(s) for s in img.shape[:3])
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    shape_ok = (shape == SRI24_SHAPE)
    spacing_ok = all(abs(z - 1.0) <= SRI24_SPACING_TOL for z in zooms)
    return shape_ok and spacing_ok


def find_channel_file(case_dir: Path, case_id: str, suffix: str) -> Path:
    """case_dir 안에서 {case}-{suffix}.nii.gz 를 찾는다."""
    cand = case_dir / f"{case_id}-{suffix}.nii.gz"
    if cand.is_file():
        return cand
    # 혹시 다른 명명: 폴더 안에서 suffix 로 끝나는 것 검색
    hits = sorted(case_dir.glob(f"*{suffix}.nii.gz"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{case_id}: channel '{suffix}' not found in {case_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="BraTS /input (케이스별 하위폴더)")
    ap.add_argument("--nnunet_in", required=True, type=Path,
                    help="nnUNet 입력을 모을 폴더 (flat, /tmp 아래)")
    ap.add_argument("--det_data", required=True, type=Path,
                    help="$det_data. <task>/raw_splitted/imagesTs 가 만들어짐")
    ap.add_argument("--task_native", default="Task002_Native")
    ap.add_argument("--task_sri24", default="Task003_SRI24")
    ap.add_argument("--map_out", required=True, type=Path,
                    help="case->task 매핑 JSON 출력 경로")
    args = ap.parse_args()

    args.nnunet_in.mkdir(parents=True, exist_ok=True)

    # 케이스 폴더 목록
    case_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"no case folders in {args.input_dir}")

    print(f"[prepare] {len(case_dirs)} cases under {args.input_dir}")

    case_to_task: dict[str, str] = {}

    for case_dir in case_dirs:
        case_id = case_dir.name  # e.g. BraTS-MET-12345-000

        # t1c 로 공간 판정
        t1c = find_channel_file(case_dir, case_id, "t1c")
        task = args.task_sri24 if is_sri24(t1c) else args.task_native
        case_to_task[case_id] = task

        # nndet imagesTs 폴더 (task 별)
        det_imagesTs = args.det_data / task / "raw_splitted" / "imagesTs"
        det_imagesTs.mkdir(parents=True, exist_ok=True)

        # 4채널 복사 (nnUNet + nndet 둘 다)
        for suffix, ch in CHANNEL_SUFFIX.items():
            src = find_channel_file(case_dir, case_id, suffix)
            dst_nnunet = args.nnunet_in / f"{case_id}_{ch}.nii.gz"
            dst_nndet = det_imagesTs / f"{case_id}_{ch}.nii.gz"
            shutil.copyfile(src, dst_nnunet)
            shutil.copyfile(src, dst_nndet)

        print(f"  [{case_id}] -> {task}")

    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.map_out, "w") as f:
        json.dump(case_to_task, f, indent=2)

    # 요약
    n_sri = sum(1 for t in case_to_task.values() if t == args.task_sri24)
    n_nat = len(case_to_task) - n_sri
    print(f"[prepare] done. SRI24={n_sri}, Native={n_nat}")
    print(f"[prepare] map -> {args.map_out}")


if __name__ == "__main__":
    main()