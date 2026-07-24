# ============================================================
# BraTS-MET 2026 제출용 통합 이미지
#   base conda env  : nnDetection (torch 1.11, do_seg=True 수정본, _C sm_75;8.0;8.6)
#   nnunetv2 env    : torch 2.4.1+cu121 + nnunetv2 2.8.0
#
#   /input (read-only) -> run.sh -> /output (flat)
# ============================================================
FROM nvcr.io/nvidia/pytorch:21.11-py3


#####################################################################
# => thread 동적 할당 코드
# ARG env_det_verbose=1

# # ---- nnDetection 환경변수 ----
# #  det_num_threads 는 의도적으로 여기서 고정하지 않는다 (run.sh 에서 export).
# #  폴백으로 4 만 박아둔다: run.sh 가 어떤 이유로 export 못 해도 과구독 안 나게.
# ENV det_data=/opt/data \
#     det_models=/opt/models \
#     det_num_threads=4 \
#     det_verbose=$env_det_verbose \
#     OMP_NUM_THREADS=1
#####################################################################
ARG env_det_num_threads=6
ARG env_det_verbose=1

# ---- nnDetection 환경변수 ----
ENV det_data=/opt/data \
    det_models=/opt/models \
    det_num_threads=$env_det_num_threads \
    det_verbose=$env_det_verbose \
    OMP_NUM_THREADS=1
#####################################################################
# ---- 시스템 패키지 ----
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y \
    git cmake make wget gnupg build-essential software-properties-common gdb ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy \
    && pip install --upgrade requests urllib3

# ============================================================
# 1) base env: nnDetection (본인 수정 소스 COPY, do_seg=True 포함)
# ============================================================
RUN mkdir -p ${det_data} ${det_models}

# 수정한 nnDetection 소스를 통째로 복사 (do_seg=True 반영본)
COPY nnDetection/ /opt/code/nndet/
WORKDIR /opt/code/nndet

# mlflow 제거 (protobuf 충돌 회피) 후 의존성 설치
RUN sed -i "/mlflow/d" requirements.txt \
    && pip install -r requirements.txt \
    && pip install hydra-core --upgrade --pre \
    && pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git

# _C 컴파일 (A4000/A10G=sm_86, A100=sm_80, sm_75 도 커버)
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" pip install -v -e .
RUN pip install "wheel>=0.26" "protobuf==3.20.3"

# ============================================================
# 2) nnunetv2 env: 별도 conda 환경 (torch 2.4.1+cu121)
# ============================================================
RUN conda create -n nnunetv2 python=3.10 -y
RUN /opt/conda/envs/nnunetv2/bin/pip install \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121
RUN /opt/conda/envs/nnunetv2/bin/pip install nnunetv2==2.8.0

# nnUNet 환경변수 (추론은 nnUNet_results 만 실제 사용)
ENV nnUNet_results=/opt/ml/models/nnunet \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed
RUN mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed

# ============================================================
# 3) 모델 가중치 + 추론 코드 + run.sh
# ============================================================
# nndet 모델 -> det_models(/opt/models), nnunet 모델 -> nnUNet_results
COPY models/nndet/   /opt/models/
COPY models/nnunet/  /opt/ml/models/nnunet/
COPY src/            /opt/ml/src/
COPY run.sh          /opt/ml/run.sh
RUN chmod +x /opt/ml/run.sh

# ---- 권한: root / --user 2061 둘 다 쓰기 가능하도록 ----
RUN chmod -R 777 /opt/data /opt/models /opt/ml /tmp/nnunet_raw /tmp/nnunet_preprocessed

WORKDIR /opt/ml
ENTRYPOINT ["/opt/ml/run.sh"]