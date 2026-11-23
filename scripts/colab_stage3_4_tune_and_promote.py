# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 3 & 4 Fine-Tuning, Mixed ID+OOD L-BFGS Calibration & Promotion Gate on Colab L4 GPU."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _ensure_deps() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"]
    )


def pre_tokenize_lean_canonical(text: str) -> list[str]:
    """Mirror Rust `lean_sys1_ort::pre_tokenize_lean_canonical` identically."""
    tokens: list[str] = []
    for word in text.split():
        i = 0
        cur: list[str] = []
        n = len(word)
        while i < n:
            if i + 2 < n and word[i : i + 3] == "[_]":
                if cur:
                    tokens.append("".join(cur))
                    cur.clear()
                tokens.append("[_]")
                i += 3
                continue
            if i + 1 < n and word[i : i + 2] in ("::", "++", "->", "=>", "<=", ">=", "!=", ":="):
                if cur:
                    tokens.append("".join(cur))
                    cur.clear()
                tokens.append(word[i : i + 2])
                i += 2
                continue
            ch = word[i]
            if ch in "()[]{},.;:":
                if cur:
                    tokens.append("".join(cur))
                    cur.clear()
                tokens.append(ch)
                i += 1
                continue
            cur.append(ch)
            i += 1
        if cur:
            tokens.append("".join(cur))
    return tokens


def encode_canonical_text(
    vocab: dict[str, int], text: str, max_len: int = 512
) -> tuple[list[int], list[int]]:
    """Mirror Rust `lean_sys1_ort::encode_canonical_text` identically."""
    unk_id = vocab.get("<unk>", 1)
    pad_id = vocab.get("<pad>", 0)
    bos_id = vocab.get("<s>", 2)
    eos_id = vocab.get("</s>", 3)

    content_ids: list[int] = []
    for tok in pre_tokenize_lean_canonical(text):
        if tok in vocab:
            content_ids.append(vocab[tok])
            continue
        start = 0
        while start < len(tok):
            matched_id: int | None = None
            matched_end = start + 1
            for end in range(len(tok), start, -1):
                sub = tok[start:end]
                if sub in vocab:
                    matched_id = vocab[sub]
                    matched_end = end
                    break
            content_ids.append(matched_id if matched_id is not None else unk_id)
            start = matched_end

    ids = [bos_id, *content_ids[: max(0, max_len - 2)]]
    if len(ids) < max_len:
        ids.append(eos_id)
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        mask.append(0)
    return ids, mask


def build_curriculum_samples(rng: random.Random) -> list[tuple[str, int, float, int]]:
    """Generate diverse canonicalized Lean 4 states covering all 5 StrategyChoice classes."""
    samples: list[tuple[str, int, float, int]] = []
    atomic_templates = [
        "_v0 : Nat ⊢ _v0 + 0 = _v0",
        "_v0 : Nat, _v1 : Nat ⊢ _v0 + _v1 = _v1 + _v0",
        "_v0 : List Nat, _v1 : List Nat ⊢ (_v0 ++ _v1).length = _v0.length + _v1.length",
        "_v0 : Nat, _v1 : Nat, _v2 : _v0 ≤ _v1 ⊢ _v0 + 1 ≤ _v1 + 1",
        "_v0 : Nat, _v1 : Nat, _v2 : _v0 < 100, _v3 : _v1 < 100 ⊢ _v0 + _v1 < 2 * 100",
        "_v0 : LWWReg ⊢ merge _v0 _v0 = _v0",
        "_v0 : LWWReg, _v1 : LWWReg ⊢ merge _v0 _v1 = merge _v1 _v0",
        "_v0 : Nat, _v1 : Nat, _v2 : Nat, _v3 : _v0 + _v1 = _v2 ⊢ (_v0 + _v1) % 256 = _v2 % 256",
    ]
    rewrite_templates = [
        "_v0 : List \u03b1 ⊢ fastRev (fastRev _v0) = _v0",
        "_v0 : Nat, _v1 : Op ⊢ evalOpt (.add (.lit _v0) _v1) = some (_v0 + evalVal _v1)",
        "_v0 : Expr ⊢ evalOpt _v0 = some (evalTotal _v0)",
        "_v0 : Nat ⊢ doubleIt (doubleIt _v0) = 4 * _v0",
        "_v0 : Box ⊢ idBox _v0 = _v0",
    ]
    apply_templates = [
        "_v0 : Nat, _v1 : Nat, _v2 : Nat ⊢ (_v0 + _v1) + _v2 = _v0 + (_v1 + _v2)",
        "_v0 : Nat, _v1 : Nat ⊢ _v0 * _v1 = _v1 * _v0",
        (
            "_v0 : List Nat, _v1 : List Nat, _v2 : List Nat ⊢ "
            "(_v0 ++ _v1) ++ _v2 = _v0 ++ (_v1 ++ _v2)"
        ),
        "_v0 : Nat ⊢ 0 + _v0 = _v0",
    ]
    have_templates = [
        "_v0 : Prop, _v1 : Prop, _v2 : _v0 ∧ _v1 ⊢ _v1 ∧ _v0",
        "_v0 : Nat, _v1 : ∃ k, _v0 = 2 * k ⊢ ∃ m, _v0 * _v0 = 2 * m",
        "_v0 : Prop, _v1 : Prop, _v2 : _v0 → _v1, _v3 : _v0 ⊢ _v1",
        "_v0 : Nat, _v1 : ∃ k, _v0 = 2 * k + 1 ⊢ ∃ m, _v0 + 1 = 2 * m",
    ]
    induction_templates = [
        "_v0 : List \u03b1, _v1 : List \u03b1 ⊢ revAcc _v0 _v1 = _v0.reverse ++ _v1",
        (
            "_v0 : List Instr, _v1 : List Instr, _v2 : List Nat ⊢ "
            "exec (_v0 ++ _v1) _v2 = exec _v1 (exec _v0 _v2)"
        ),
        "_v0 : Expr, _v1 : List Nat ⊢ exec (compile _v0) _v1 = eval _v0 :: _v1",
        "_v0 : BTree \u03b1 ⊢ mirror (mirror _v0) = _v0",
        "_v0 : BTree \u03b1 ⊢ size (mirror _v0) = size _v0",
        "_v0 : BinTree \u03b1 ⊢ flipTree (flipTree _v0) = _v0",
        (
            "_v0 : List Op, _v1 : List Op, _v2 : List Nat ⊢ "
            "stepAll (_v0 ++ _v1) _v2 = stepAll _v1 (stepAll _v0 _v2)"
        ),
    ]

    groups = [
        (0, atomic_templates, 1.0, 0),
        (1, rewrite_templates, 0.7, 1),
        (2, apply_templates, 0.4, 1),
        (3, have_templates, 0.3, 2),
        (4, induction_templates, 0.2, 2),
    ]
    for _ in range(140):
        for strat, tmpls, noul, diff in groups:
            samples.append((rng.choice(tmpls), strat, noul, diff))
    rng.shuffle(samples)
    return samples


def build_ood_calibration_samples(rng: random.Random) -> list[tuple[str, int, float, int]]:
    """OOD calibration split covering multi-struct + List compositions and open conjectures."""
    ood_templates = [
        ("_v0 : Ledger, _v1 : List Tx ⊢ replay _v0 _v1 = applyAll _v1 _v0", 1, 0.5, 1),
        ("_v0 : Action, _v1 : State ⊢ step _v0 _v1 = nextState _v0 _v1", 0, 0.8, 0),
        (
            "_v0 : List Cmd, _v1 : List Cmd, _v2 : State ⊢ "
            "runAll (_v0 ++ _v1) _v2 = runAll _v1 (runAll _v0 _v2)",
            4,
            0.2,
            2,
        ),
        (
            "_v0 : Nat, _v1 : Nat, _v2 : _v0 ≤ _v1 ⊢ "
            "∃ _v3 _v4 : Nat, _v3 + _v4 = _v0 + _v1 ∧ _v3 = _v0 - 1",
            3,
            0.2,
            2,
        ),
        # Open-conjecture OOD probes (spread across classes to penalize overconfidence)
        ("_v0 : Nat ⊢ ∃ _v1 _v2 : Nat, PrimeGap _v1 _v2 ∧ _v0 < _v1", 2, 0.1, 2),
        ("_v0 : Nat ⊢ ∃ _v1 _v2 : Nat, PrimeGap _v1 _v2 ∧ _v0 < _v1", 3, 0.1, 2),
    ]
    out: list[tuple[str, int, float, int]] = []
    for _ in range(30):
        out.extend(ood_templates)
    rng.shuffle(out)
    return out


def _assert_premise_fit_initializers(onnx_path: Path) -> None:
    import numpy as np
    import onnx

    model = onnx.load(str(onnx_path))
    init_map = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    required = {
        "heads.premise_fit.0.weight": (384, 1536),
        "heads.premise_fit.0.bias": (384,),
        "heads.premise_fit.2.weight": (1, 384),
        "heads.premise_fit.2.bias": (1,),
    }
    for name, expected_shape in required.items():
        if name not in init_map:
            raise AssertionError(f"Missing required ONNX initializer: {name}")
        arr = init_map[name]
        if tuple(arr.shape) != expected_shape:
            raise AssertionError(
                f"Shape mismatch for {name}: expected {expected_shape}, got {arr.shape}"
            )
        if "weight" in name and float(np.var(arr)) <= 1e-8:
            raise AssertionError(f"Degenerate zero-variance weights in {name}")


def _resolve_tokenizer_path() -> Path:
    for candidate in (
        Path("crates/lean-prover/assets/tokenizer.json"),
        Path("/content/tokenizer.json"),
        Path("artifacts/tokenizer.json"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate tokenizer.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3/4 Colab L4 training, OOD calibration, and ONNX promotion."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("crates/lean-prover/assets"),
        help="Output directory for lean_sys1.onnx and temperatures.json",
    )
    parser.add_argument("--epochs", type=int, default=28, help="Training epochs")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for local smoke checks (Colab L4 GPU by default)",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    _ensure_deps()

    import numpy as np
    import onnx
    import onnxruntime as ort  # type: ignore[import-untyped]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from onnx import numpy_helper

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Stage 3 & 4 require an L4/GPU runtime (pass --allow-cpu for dry-run).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = _resolve_tokenizer_path()
    vocab: dict[str, int] = json.loads(tok_path.read_text(encoding="utf-8"))["model"]["vocab"]
    print(f"[Stage 3/4] device={device} tokenizer={tok_path} out_dir={args.out_dir}")

    vocab_size, d_model, n_heads, ffn_dim, max_len = 8192, 384, 6, 1024, 512
    head_dim = d_model // n_heads
    seq_len = 128

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight

    class LeanSys1OnnxModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc: Any = nn.Module()
            self.enc.tok_emb = nn.Embedding(vocab_size, d_model)
            block: Any = nn.Module()
            block.norm1 = RMSNorm(d_model)
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            block.attn.proj = nn.Linear(d_model, d_model, bias=False)
            block.norm2 = RMSNorm(d_model)
            block.ffn = nn.Module()
            block.ffn.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
            block.ffn.w_up = nn.Linear(d_model, ffn_dim, bias=False)
            block.ffn.w_down = nn.Linear(ffn_dim, d_model, bias=False)
            self.enc.blocks = nn.ModuleList([block])
            self.enc.norm_out = RMSNorm(d_model)
            self.enc.pool_query = nn.Linear(d_model, 1, bias=False)

            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            pos = torch.arange(max_len).float().unsqueeze(1)
            freqs = torch.cat([pos * inv_freq, pos * inv_freq], dim=-1)
            self.enc.register_buffer("rope_cos", freqs.cos())
            self.enc.register_buffer("rope_sin", freqs.sin())

            self.heads: Any = nn.Module()
            self.heads.choice_head = nn.Linear(d_model, 5)
            self.heads.noul_head = nn.Linear(d_model, 1)
            self.heads.score_head = nn.Linear(d_model, 3)
            self.heads.premise_fit = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            bsz, seq = input_ids.shape
            x = self.enc.tok_emb(input_ids)
            blk = self.enc.blocks[0]
            h1 = blk.norm1(x)
            qkv = blk.attn.qkv(h1).view(bsz, seq, 3, n_heads, head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            cos = self.enc.rope_cos[:seq].unsqueeze(0).unsqueeze(0)
            sin = self.enc.rope_sin[:seq].unsqueeze(0).unsqueeze(0)
            half = head_dim // 2
            q_rot = q * cos + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * sin
            k_rot = k * cos + torch.cat([-k[..., half:], k[..., :half]], dim=-1) * sin
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_out = (
                torch.matmul(F.softmax(scores, dim=-1), v)
                .transpose(1, 2)
                .contiguous()
                .view(bsz, seq, d_model)
            )
            x2 = x + blk.attn.proj(attn_out)
            h2 = blk.norm2(x2)
            ffn_out = blk.ffn.w_down(F.silu(blk.ffn.w_gate(h2)) * blk.ffn.w_up(h2))
            h_out = self.enc.norm_out(x2 + ffn_out)
            pool_scores = (
                self.enc.pool_query(h_out)
                .squeeze(-1)
                .masked_fill(attention_mask == 0, float("-inf"))
            )
            state_emb = (h_out * F.softmax(pool_scores, dim=-1).unsqueeze(-1)).sum(dim=1)
            return (
                state_emb,
                self.heads.choice_head(state_emb),
                self.heads.noul_head(state_emb),
                self.heads.score_head(state_emb),
            )

        def score_premise_fit(self, s_x: torch.Tensor, e_l: torch.Tensor) -> torch.Tensor:
            """Match Rust `score_premise_fit_batch` 4d feature."""
            feat = torch.cat([s_x, e_l, torch.abs(s_x - e_l), s_x * e_l], dim=-1)
            return self.heads.premise_fit(feat)

    rng = random.Random(42)
    samples = build_curriculum_samples(rng)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] + build_ood_calibration_samples(rng)

    def _to_tensors(
        batch: list[tuple[str, int, float, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_ids, all_mask, all_y, all_noul = [], [], [], []
        for text, y, noul, _ in batch:
            ids, mask = encode_canonical_text(vocab, text, seq_len)
            all_ids.append(ids)
            all_mask.append(mask)
            all_y.append(y)
            all_noul.append([noul])
        return (
            torch.tensor(all_ids, dtype=torch.long, device=device),
            torch.tensor(all_mask, dtype=torch.long, device=device),
            torch.tensor(all_y, dtype=torch.long, device=device),
            torch.tensor(all_noul, dtype=torch.float32, device=device),
        )

    tr_ids, tr_mask, tr_y, tr_noul = _to_tensors(train_samples)
    va_ids, va_mask, va_y, _ = _to_tensors(val_samples)

    model = LeanSys1OnnxModule().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-2, fused=(device.type == "cuda")
    )
    lambda_sigreg = 0.02
    counts = torch.bincount(tr_y, minlength=5).float().clamp(min=1.0)
    class_weights = len(tr_y) / (5.0 * counts)
    class_weights = class_weights / class_weights.mean()

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch_size = 256
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(tr_y), device=device)
        ce_loss = torch.zeros((), device=device)
        sigreg_loss = torch.zeros((), device=device)
        for i in range(0, len(tr_y), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                s_x, choice_logits, noul_logit, _ = model(tr_ids[idx], tr_mask[idx])
                ce_loss = F.cross_entropy(
                    choice_logits,
                    tr_y[idx],
                    weight=class_weights,
                    label_smoothing=0.075,
                )
                noul_loss = F.binary_cross_entropy_with_logits(noul_logit, tr_noul[idx])
                sigreg_loss = s_x.mean(dim=0).pow(2).mean() + (s_x.var(dim=0) - 1.0).pow(2).mean()
                s_shift = torch.roll(s_x, shifts=1, dims=0)
                y_shift = torch.roll(tr_y[idx], shifts=1, dims=0)
                fit_logit = model.score_premise_fit(s_x, s_shift)
                fit_target = (tr_y[idx] == y_shift).float().unsqueeze(-1)
                fit_loss = F.binary_cross_entropy_with_logits(fit_logit, fit_target)
                loss = ce_loss + 0.5 * noul_loss + 0.35 * fit_loss + lambda_sigreg * sigreg_loss
            loss.backward()
            opt.step()
        if epoch % 7 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                _, v_logits, _, _ = model(va_ids, va_mask)
                acc = (v_logits.argmax(dim=-1) == va_y).float().mean().item()
            print(
                f"[Stage 3] epoch={epoch}/{args.epochs} ce={ce_loss.item():.4f} "
                f"sigreg={sigreg_loss.item():.4f} val_acc={acc:.4f}"
            )

    model.eval()
    print("[Stage 3] Running L-BFGS temperature calibration on mixed ID+OOD split [0.5, 2.5]")
    with torch.no_grad():
        _, val_logits, _, _ = model(va_ids, va_mask)
    t_choice = nn.Parameter(torch.ones(1, device=device))
    lbfgs = torch.optim.LBFGS([t_choice], lr=0.1, max_iter=50)

    def _cal_closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = F.cross_entropy(val_logits / t_choice.clamp(0.5, 2.5), va_y)
        loss.backward()
        return loss

    lbfgs.step(_cal_closure)
    t_choice_val = float(torch.clamp(t_choice.detach(), 0.5, 2.5).item())

    # Smoke-check OOD confidence on open conjecture
    with torch.no_grad():
        ood_ids, ood_mask = encode_canonical_text(
            vocab, "_v0 : Nat ⊢ ∃ _v1 _v2 : Nat, PrimeGap _v1 _v2 ∧ _v0 < _v1", seq_len
        )
        _, ood_logits, _, _ = model(
            torch.tensor([ood_ids], dtype=torch.long, device=device),
            torch.tensor([ood_mask], dtype=torch.long, device=device),
        )
        ood_probs = F.softmax(ood_logits[0] / t_choice_val, dim=-1)
        ood_conf = float(ood_probs.max().item())
        print(
            f"[Stage 3] Open-conjecture OOD max_conf={ood_conf:.4f} (T_choice={t_choice_val:.4f})"
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu().eval()
    dummy_ids = torch.zeros((1, 512), dtype=torch.long)
    dummy_mask = torch.ones((1, 512), dtype=torch.long)
    onnx_path = out_dir / "lean_sys1.onnx"
    torch.onnx.export(
        cpu_model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["state_emb", "choice_logits", "noul_logit", "score_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    existing_inits = {init.name for init in onnx_model.graph.initializer}
    state_dict = cpu_model.state_dict()
    for name in (
        "heads.premise_fit.0.weight",
        "heads.premise_fit.0.bias",
        "heads.premise_fit.2.weight",
        "heads.premise_fit.2.bias",
    ):
        if name not in existing_inits:
            arr = state_dict[name].detach().cpu().numpy()
            onnx_model.graph.initializer.append(numpy_helper.from_array(arr, name=name))
    onnx.save(onnx_model, str(onnx_path))
    _assert_premise_fit_initializers(onnx_path)

    temps = {"T_choice": t_choice_val, "T_noul": 1.0, "T_fit": 1.0}
    temp_path = out_dir / "temperatures.json"
    temp_path.write_text(json.dumps(temps, indent=2), encoding="utf-8")

    # ORT CPU verification + Rust parity reference print
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref_ids = np.zeros((1, 512), dtype=np.int64)
    ref_mask = np.zeros((1, 512), dtype=np.int64)
    for i, tok in enumerate((2, 10, 15, 3)):
        ref_ids[0, i] = tok
        ref_mask[0, i] = 1
    t0 = time.perf_counter()
    ort_out = sess.run(None, {"input_ids": ref_ids, "attention_mask": ref_mask})
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    state_arr = np.asarray(ort_out[0], dtype=np.float32)
    choice_arr = np.asarray(ort_out[1], dtype=np.float32)
    state_prefix = [round(float(x), 7) for x in state_arr[0][:8]]
    choice_ref = [round(float(x), 7) for x in choice_arr[0]]
    print(f"[Stage 4] Verified {onnx_path} (cpu_ms={cpu_ms:.2f}, T_choice={t_choice_val:.4f})")
    print(f"[Stage 4] Rust parity ref_state_prefix={state_prefix}")
    print(f"[Stage 4] Rust parity ref_choice={choice_ref}")


if __name__ == "__main__":
    main()
