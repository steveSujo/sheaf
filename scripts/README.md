# Reproducing Model Training & Calibration

End users building and running sheaf do **not** need Python, because `lean_sys1.onnx`, `tokenizer.json`, `temperatures.json`, and `mathlib_sq8.vdb` are embedded directly into the Rust binary. These scripts reproduce the PyTorch training, OOD temperature calibration, and ONNX export pipeline from scratch.

## Hardware & Environment

- **Target GPU:** NVIDIA L4 (24 GB VRAM) or any CUDA GPU supporting bfloat16 (`--allow-cpu` is available on `scripts/colab_stage3_4_tune_and_promote.py` for smoke checks).
- **Dependencies:** Install the pinned packages from `scripts/requirements-train.txt`:

```bash
pip install -r scripts/requirements-train.txt
```

## 1. Stage 1 Pre-Training (`scripts/colab_stage1_train.py`)

Trains LeanSys1OnnxModule (RoPE + RMSNorm + SwiGLU + masked attention pooling + 4d PremiseFit MLP) on the 5-class StrategyChoice curriculum with SIGReg regularization and label smoothing (0.075), then runs L-BFGS calibration and exports `lean_sys1.onnx` and `temperatures.json`.

```bash
python scripts/colab_stage1_train.py
```

- **Default output:** `artifacts/lean_sys1.onnx` and `artifacts/temperatures.json` (or `/content/artifacts/` on Colab).

## 2. Stage 3/4 Fine-Tuning & OOD Calibration (`scripts/colab_stage3_4_tune_and_promote.py`)

Fine-tunes LeanSys1OnnxModule with label smoothing (0.075) and PremiseFit BCE loss, calibrates T_choice via L-BFGS on a mixed in-distribution and OOD split (producing T_choice = 1.6071), injects the PremiseFit weights into the exported ONNX graph, and verifies parity with onnxruntime.

```bash
python scripts/colab_stage3_4_tune_and_promote.py --out-dir crates/lean-prover/assets
```

- **Expected output:**
  - `crates/lean-prover/assets/lean_sys1.onnx` with verified PremiseFit initializers.
  - `crates/lean-prover/assets/temperatures.json` with calibrated T_choice in [0.5, 2.5].
  - Printed state prefix and choice reference vectors for `cargo test -p lean-sys1-ort --release` parity verification.
