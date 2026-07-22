"""
nnUNet inference + post-processing driver (BraTS-METS final submission).

Reads configs/nnunet.yaml and runs one of three modes:

  A_ONLY : predict A (RC specialist) -> remap (MODEL_A.REMAP) -> optional postprocess
  B_ONLY : predict B (primary)                                -> optional postprocess
  MERGE  : predict A -> remap A (MERGE.REMAP, ALWAYS) -> [optional post A]
           predict B                                          -> [optional post B]
           merge A into B (B-priority)   [final = merged; no post-merge postprocess]

All external steps use the existing CLI tools so the exact same code paths run:
  * prediction  -> `nnUNetv2_predict`
  * remap       -> post_processing/nnunet/remap_rc_pred_to_challenge.py
  * merge       -> post_processing/nnunet/merge_rc_b_priority.py
  * postprocess -> post_processing/nnunet/postprocess_5mm.py

Every stage writes to its own subfolder under OUTPUT_BASE.

Usage:
    python tests/nnunet_test.py --config configs/nnunet.yaml
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]                 # for_upload/
PP_DIR = REPO_ROOT / "post_processing" / "nnunet"
REMAP_SCRIPT = PP_DIR / "remap_rc.py"
MERGE_SCRIPT = PP_DIR / "merge_rc.py"
POSTPROC_SCRIPT = PP_DIR / "remove_component.py"


def run(cmd, env=None):
    print("\n$ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def build_env(cfg):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("DEVICE", "0"))
    for k, v in (cfg.get("ENV") or {}).items():
        if v:
            env[k] = str(v)
    return env


def predict(model_cfg, input_dir, output_dir, predict_cfg, env):
    """Run `nnUNetv2_predict` for one model into output_dir."""
    folds = [str(f) for f in model_cfg["FOLDS"]]
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(model_cfg["DATASET"]),
        "-p", model_cfg["PLANS"],
        "-tr", model_cfg["TRAINER"],
        "-c", model_cfg["CONFIG"],
        "-f", *folds,
        "-chk", model_cfg["CHECKPOINT"],
        "-step_size", str(predict_cfg.get("STEP_SIZE", 0.5)),
        "-npp", str(predict_cfg.get("NPP", 2)),
        "-nps", str(predict_cfg.get("NPS", 2)),
    ]
    if predict_cfg.get("DISABLE_TTA", False):
        cmd.append("--disable_tta")
    run(cmd, env=env)


def remap(input_dir, output_dir, remap_cfg, env):
    cmd = [sys.executable, REMAP_SCRIPT,
           "--input-dir", input_dir,
           "--output-dir", output_dir,
           "--mode", remap_cfg.get("SCHEME", "binary"),
           "--rc-label", str(remap_cfg.get("RC_LABEL", 4))]
    run(cmd, env=env)


def merge(a_dir, b_dir, output_dir, merge_cfg, env):
    cmd = [sys.executable, MERGE_SCRIPT,
           "--a-dir", a_dir,
           "--b-dir", b_dir,
           "--output-dir", output_dir,
           "--rc-label", str(merge_cfg.get("RC_LABEL", 4)),
           "--overlap-mode", merge_cfg.get("OVERLAP_MODE", "component")]
    run(cmd, env=env)


def _per_class_str(pp_cfg):
    """Build 'label:mm3,...' from a POSTPROCESS.MIN_VOL_MM3 mapping (keys are label names)."""
    d = pp_cfg.get("MIN_VOL_MM3") or {}
    return ",".join(f"{k}:{float(v)}" for k, v in d.items())


def postprocess(input_dir, output_dir, pp_cfg, env):
    cmd = [sys.executable, POSTPROC_SCRIPT,
           "--input", input_dir,
           "--output", output_dir,
           "--default-min-vol", str(pp_cfg.get("DEFAULT_MIN_VOL", 0))]
    per_class = _per_class_str(pp_cfg)
    if per_class:
        cmd += ["--per-class", per_class]
    run(cmd, env=env)


def maybe_postprocess(src_dir, out_dir, pp_cfg, env):
    """Run postprocess if pp_cfg.ENABLED; return the folder holding the result."""
    if pp_cfg and pp_cfg.get("ENABLED", False):
        postprocess(src_dir, out_dir, pp_cfg, env)
        return out_dir
    return src_dir


def main():
    ap = argparse.ArgumentParser(description="nnUNet inference + post-processing driver.")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "nnunet.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mode = cfg["MODE"].upper()
    env = build_env(cfg)
    input_dir = cfg["INPUT_DIR"]
    out = Path(cfg["OUTPUT_BASE"])
    out.mkdir(parents=True, exist_ok=True)
    predict_cfg = cfg.get("PREDICT", {})

    print(f"=== nnUNet driver | MODE={mode} | OUTPUT_BASE={out} ===")

    if mode == "A_ONLY":
        a_cfg = cfg["MODEL_A"]
        a_pred = str(out / "model_a_pred")
        predict(a_cfg, input_dir, a_pred, predict_cfg, env)

        last = a_pred
        rcfg = a_cfg.get("REMAP", {})
        if rcfg.get("ENABLED", False):
            a_remap = str(out / "model_a_remap")
            remap(last, a_remap, rcfg, env)
            last = a_remap

        last = maybe_postprocess(last, str(out / "model_a_post"), a_cfg.get("POSTPROCESS", {}), env)
        print(f"\n=== A_ONLY done. Final: {last} ===")

    elif mode == "B_ONLY":
        b_cfg = cfg["MODEL_B"]
        b_pred = str(out / "model_b_pred")
        predict(b_cfg, input_dir, b_pred, predict_cfg, env)   # B is already challenge labels (no remap)

        last = maybe_postprocess(b_pred, str(out / "model_b_post"), b_cfg.get("POSTPROCESS", {}), env)
        print(f"\n=== B_ONLY done. Final: {last} ===")

    elif mode == "MERGE":
        mcfg = cfg["MERGE"]

        # 1) predict A -> 2) remap A (ALWAYS) -> 3) optional post A
        a_pred = str(out / "model_a_pred")
        predict(cfg["MODEL_A"], input_dir, a_pred, predict_cfg, env)
        a_remap = str(out / "model_a_remap")
        remap(a_pred, a_remap, mcfg.get("REMAP", {}), env)
        a_for_merge = maybe_postprocess(a_remap, str(out / "model_a_post"),
                                        mcfg.get("POSTPROCESS_A", {}), env)

        # 4) predict B -> 5) optional post B
        b_pred = str(out / "model_b_pred")
        predict(cfg["MODEL_B"], input_dir, b_pred, predict_cfg, env)
        b_for_merge = maybe_postprocess(b_pred, str(out / "model_b_post"),
                                        mcfg.get("POSTPROCESS_B", {}), env)

        # 6) merge A into B (B-priority). The merged result is the final output
        #    (post-processing is applied per model BEFORE merge, a_post+b_post style).
        merged = str(out / "merged")
        merge(a_for_merge, b_for_merge, merged, mcfg, env)

        print(f"\n=== MERGE done. Final: {merged} ===")

    else:
        raise ValueError(f"Unknown MODE '{mode}'. Use one of: A_ONLY, B_ONLY, MERGE.")


if __name__ == "__main__":
    main()
