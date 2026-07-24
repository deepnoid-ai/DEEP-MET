#!/bin/bash
# ============================================================
# BraTS-MET 2026 submission entrypoint
#   /input (read-only) -> /output (flat)
#
#   1) prepare_input.py : size 기준 Task002/003 분류 + 입력 배치
#   2) nnUNetv2_predict  : segmentation mask (nnunetv2 env)
#   3) nndet_predict     : Task002, Task003 각각 (base env, do_seg=True)
#   4) boxes_2_seg_4.py  : 두 결과 병합 -> 최종 mask
#   5) /output 로 flat 하게 복사
# ============================================================
set -euo pipefail

#####################################################################
# => thread 동적 할당 코드
# # >>> 추가: nnDetection 스레드 수 동적 설정 ----------------------
# N_CORES=$(nproc)
# DET_THREADS=$(( N_CORES > 2 ? N_CORES / 2 : 1 ))
# export det_num_threads=${DET_THREADS}
# export OMP_NUM_THREADS=1
# echo "[info] detected ${N_CORES} cores -> det_num_threads=${DET_THREADS}"
# # <<< 추가 끝 --------------------------------------------------
#####################################################################

# ---- 경로 ----
INPUT_DIR=/input
OUTPUT_DIR=/output
SRC=/opt/ml/src
WORK=/tmp/work

NNUNET_IN=${WORK}/nnunet_in
NNUNET_OUT_ALL=${WORK}/nnunet_out_all
NNUNET_OUT_RC=${WORK}/nnunet_out_rc
NNUNET_OUT=${WORK}/nnunet_out
MERGED_OUT=${WORK}/merged
MAP_JSON=${WORK}/case_to_task.json

NNUNET_BIN=/opt/conda/envs/nnunetv2/bin/nnUNetv2_predict

TASK_NATIVE=Task002_Native
TASK_SRI24=Task003_SRI24
NNDET_MODEL=RetinaUNetV001_D3V001_3d

mkdir -p "${NNUNET_IN}" "${NNUNET_OUT}" "${MERGED_OUT}" "${OUTPUT_DIR}" "${NNUNET_OUT_ALL}" "${NNUNET_OUT_RC}"

echo "================ STEP 1: prepare_input ================"
python "${SRC}/prepare_input.py" \
    --input_dir "${INPUT_DIR}" \
    --nnunet_in "${NNUNET_IN}" \
    --det_data  "${det_data}" \
    --task_native "${TASK_NATIVE}" \
    --task_sri24  "${TASK_SRI24}" \
    --map_out "${MAP_JSON}"

echo "================ STEP 2-1: nnUNetv2 predict All ================"
"${NNUNET_BIN}" \
    -i "${NNUNET_IN}" -o "${NNUNET_OUT_ALL}" \
    -d 001 -c 3d_fullres -f all \
    -p nnUNetResEncUNetXLPlans -tr nnUNetTrainer \
    -npp 2 -nps 2

echo "================ STEP 2-2: nnUNetv2 predict RC ================"
"${NNUNET_BIN}" \
    -i "${NNUNET_IN}" -o "${NNUNET_OUT_RC}" \
    -d 002 -c 3d_fullres -f all \
    -p nnUNetResEncUNetXLPlans -tr nnUNetTrainer \
    -npp 2 -nps 2

echo "================ STEP 3: remap_RC ================"
python "${SRC}/postprocess/remap_rc.py" \
    --input-dir "${NNUNET_OUT_RC}" \
    --output-dir "${NNUNET_OUT_RC}" \
    --mode  "binary"

echo "================ STEP 3-1: postprocess RC ================"
python "${SRC}/postprocess/postprocess_5mm.py" \
    --input  "${NNUNET_OUT_RC}" \
    --output "${NNUNET_OUT_RC}" \
    --per-class "RC:5" \
    --default-min-vol 0

echo "================ STEP 3-2: postprocess All ================"
python "${SRC}/postprocess/postprocess_5mm.py" \
    --input  "${NNUNET_OUT_ALL}" \
    --output "${NNUNET_OUT_ALL}" \
    --per-class "NETC:5,SNFH:5,ET:5,RC:5" \
    --default-min-vol 0

echo "================ STEP 4: merge_RC ================"
python "${SRC}/postprocess/merge_rc.py" \
    --a-dir "${NNUNET_OUT_RC}" \
    --b-dir "${NNUNET_OUT_ALL}" \
    --output-dir  "${NNUNET_OUT}"

echo "================ STEP 5: nnDetection predict ================"
# Task 별 imagesTs 에 케이스가 있으면 그 Task 추론
for TASK_NUM in 002 003; do
    if [ "${TASK_NUM}" = "002" ]; then TASK_NAME=${TASK_NATIVE}; else TASK_NAME=${TASK_SRI24}; fi
    IMAGES_TS="${det_data}/${TASK_NAME}/raw_splitted/imagesTs"
    if [ -d "${IMAGES_TS}" ] && [ -n "$(ls -A "${IMAGES_TS}" 2>/dev/null)" ]; then
        echo "  -> nndet_predict ${TASK_NUM} (${TASK_NAME})"
        nndet_predict "${TASK_NUM}" "${NNDET_MODEL}" --num_tta 4
    else
        echo "  -> skip ${TASK_NUM} (no cases)"
    fi
done

echo "================ STEP 6: merge (boxes_2_seg_4) ================"
# Task 별 test_predictions 폴더를 각각 merge -> 같은 MERGED_OUT 으로
# Native = score_thresh 0.15, SRI24 = score_thresh 0.2
for TASK_NAME in "${TASK_NATIVE}" "${TASK_SRI24}"; do
    # Task 별 score_thresh 결정
    if [ "${TASK_NAME}" = "${TASK_NATIVE}" ]; then
        SCORE_THRESH=0.15
    else
        SCORE_THRESH=0.2
    fi

    PRED_DIR="${det_models}/${TASK_NAME}/${NNDET_MODEL}/consolidated/test_predictions"
    if [ -d "${PRED_DIR}" ] && [ -n "$(ls -A "${PRED_DIR}"/*_boxes.pkl 2>/dev/null)" ]; then
        echo "  -> merge ${TASK_NAME} (score_thresh=${SCORE_THRESH})"
        python "${SRC}/postprocess/boxes_2_seg_4.py" \
            --pred_dir      "${PRED_DIR}" \
            --seg_dir       "${NNUNET_OUT}" \
            --output_dir    "${MERGED_OUT}" \
            --score_thresh  "${SCORE_THRESH}" \
            --et_thresh     27 \
            --small_class   0 \
            --confirm_mode  center \
            --thresh-netc 5 --thresh-snfh 5 --thresh-et 5 --thresh-rc 5
    else
        echo "  -> skip merge ${TASK_NAME} (no boxes)"
    fi
done

echo "================ STEP 7: finalize -> /output (flat) ================"
# merged 결과를 /output 으로 flat 하게
for f in "${MERGED_OUT}"/*.nii.gz; do
    [ -e "${f}" ] || continue
    cp "${f}" "${OUTPUT_DIR}/$(basename "${f}")"
done

echo "[done] outputs in ${OUTPUT_DIR}:"
ls -1 "${OUTPUT_DIR}" | head