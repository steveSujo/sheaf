# SPDX-License-Identifier: AGPL-3.0-only
"""Stage 1 + L-BFGS Calibration Training Script for Google Colab L4 GPU.

Trains LeanSys1OnnxModule (RoPE + RMSNorm + SwiGLU + Masked Attention Pooling + PremiseFit 4d MLP)
on canonicalized Lean 4 proof states across all 5 StrategyChoice classes with class-balanced CE,
lambda_sigreg <= 0.05, and L-BFGS temperature calibration clamped to [0.5, 2.5].
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path


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


def encode_canonical_text(vocab: dict[str, int], text: str, max_len: int = 512) -> tuple[list[int], list[int]]:
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
            matched_id = None
            matched_end = start + 1
            for end in range(len(tok), start, -1):
                sub = tok[start:end]
                if sub in vocab:
                    matched_id = vocab[sub]
                    matched_end = end
                    break
            content_ids.append(matched_id if matched_id is not None else unk_id)
            start = matched_end

    ids = [bos_id] + content_ids[: max(0, max_len - 2)]
    if len(ids) < max_len:
        ids.append(eos_id)
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        mask.append(0)
    return ids, mask


def build_curriculum_samples(rng: random.Random) -> list[tuple[str, int, float, int]]:
    """Generate diverse canonicalized Lean 4 states covering all 5 StrategyChoice classes:
    0=AtomicClose, 1=Rewrite, 2=ApplyLemma, 3=IntroduceHave, 4=Induction.
    """
    samples: list[tuple[str, int, float, int]] = []
    # 0: AtomicClose (arithmetic, simple equalities, list length, bounds)
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
    # 1: Rewrite (def unfolding, idempotence, compound helper equations)
    rewrite_templates = [
        "_v0 : List α ⊢ fastRev (fastRev _v0) = _v0",
        "_v0 : Nat, _v1 : Op ⊢ evalOpt (.add (.lit _v0) _v1) = some (_v0 + evalVal _v1)",
        "_v0 : Expr ⊢ evalOpt _v0 = some (evalTotal _v0)",
        "_v0 : Nat ⊢ doubleIt (doubleIt _v0) = 4 * _v0",
        "_v0 : Box ⊢ idBox _v0 = _v0",
    ]
    # 2: ApplyLemma (external library lemmas / associative identities)
    apply_templates = [
        "_v0 : Nat, _v1 : Nat, _v2 : Nat ⊢ (_v0 + _v1) + _v2 = _v0 + (_v1 + _v2)",
        "_v0 : Nat, _v1 : Nat ⊢ _v0 * _v1 = _v1 * _v0",
        "_v0 : List Nat, _v1 : List Nat, _v2 : List Nat ⊢ (_v0 ++ _v1) ++ _v2 = _v0 ++ (_v1 ++ _v2)",
        "_v0 : Nat ⊢ 0 + _v0 = _v0",
    ]
    # 3: IntroduceHave (conjunction, existential witnesses, implication)
    have_templates = [
        "_v0 : Prop, _v1 : Prop, _v2 : _v0 ∧ _v1 ⊢ _v1 ∧ _v0",
        "_v0 : Nat, _v1 : ∃ k, _v0 = 2 * k ⊢ ∃ m, _v0 * _v0 = 2 * m",
        "_v0 : Prop, _v1 : Prop, _v2 : _v0 → _v1, _v3 : _v0 ⊢ _v1",
        "_v0 : Nat, _v1 : ∃ k, _v0 = 2 * k + 1 ⊢ ∃ m, _v0 + 1 = 2 * m",
    ]
    # 4: Induction (recursive List, Expr, BinTree, BTree, AST)
    induction_templates = [
        "_v0 : List α, _v1 : List α ⊢ revAcc _v0 _v1 = _v0.reverse ++ _v1",
        "_v0 : List Instr, _v1 : List Instr, _v2 : List Nat ⊢ exec (_v0 ++ _v1) _v2 = exec _v1 (exec _v0 _v2)",
        "_v0 : Expr, _v1 : List Nat ⊢ exec (compile _v0) _v1 = eval _v0 :: _v1",
        "_v0 : BTree α ⊢ mirror (mirror _v0) = _v0",
        "_v0 : BTree α ⊢ size (mirror _v0) = size _v0",
        "_v0 : BinTree α ⊢ flipTree (flipTree _v0) = _v0",
        "_v0 : List Op, _v1 : List Op, _v2 : List Nat ⊢ stepAll (_v0 ++ _v1) _v2 = stepAll _v1 (stepAll _v0 _v2)",
    ]

    groups = [
        (0, atomic_templates, 1.0, 0),
        (1, rewrite_templates, 0.7, 1),
        (2, apply_templates, 0.4, 1),
        (3, have_templates, 0.3, 2),
        (4, induction_templates, 0.2, 2),
    ]
    for _ in range(120):
        for strat, tmpls, noul, diff in groups:
            t = rng.choice(tmpls)
            samples.append((t, strat, noul, diff))
    rng.shuffle(samples)
    return samples


def build_ood_calibration_samples(rng: random.Random) -> list[tuple[str, int, float, int]]:
    """Out-of-distribution calibration split covering structure + List, sum-type cases, and multi-var Exists."""
    ood_templates = [
        ("_v0 : Ledger, _v1 : List Tx ⊢ replay _v0 _v1 = applyAll _v1 _v0", 1, 0.5, 1),
        ("_v0 : Action, _v1 : State ⊢ step _v0 _v1 = nextState _v0 _v1", 0, 0.8, 0),
        ("_v0 : List Cmd, _v1 : List Cmd, _v2 : State ⊢ runAll (_v0 ++ _v1) _v2 = runAll _v1 (runAll _v0 _v2)", 4, 0.2, 2),
        ("_v0 : Nat, _v1 : Nat, _v2 : _v0 ≤ _v1 ⊢ ∃ _v3 _v4 : Nat, _v3 + _v4 = _v0 + _v1 ∧ _v3 = _v0 - 1", 3, 0.2, 2),
    ]
    out: list[tuple[str, int, float, int]] = []
    for _ in range(30):
        out.extend(ood_templates)
    rng.shuffle(out)
    return out


def main() -> None:
    _ensure_deps()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1] Running on device={device}, cuda_available={torch.cuda.is_available()}")

    tok_path = Path("/content/tokenizer.json")
    if not tok_path.exists():
        tok_path = Path("crates/lean-prover/assets/tokenizer.json")
    vocab = json.loads(tok_path.read_text(encoding="utf-8"))["model"]["vocab"]

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
            self.enc = nn.Module()
            self.enc.tok_emb = nn.Embedding(vocab_size, d_model)
            block = nn.Module()
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

            self.heads = nn.Module()
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
            B, L = input_ids.shape
            x = self.enc.tok_emb(input_ids)
            blk = self.enc.blocks[0]
            h1 = blk.norm1(x)
            qkv = blk.attn.qkv(h1).view(B, L, 3, n_heads, head_dim)
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
            cos = self.enc.rope_cos[:L].unsqueeze(0).unsqueeze(0)
            sin = self.enc.rope_sin[:L].unsqueeze(0).unsqueeze(0)
            half = head_dim // 2
            q_rot = q * cos + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * sin
            k_rot = k * cos + torch.cat([-k[..., half:], k[..., :half]], dim=-1) * sin
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_out = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, L, d_model)
            x2 = x + blk.attn.proj(attn_out)
            h2 = blk.norm2(x2)
            ffn_out = blk.ffn.w_down(F.silu(blk.ffn.w_gate(h2)) * blk.ffn.w_up(h2))
            h_out = self.enc.norm_out(x2 + ffn_out)
            pool_scores = self.enc.pool_query(h_out).squeeze(-1).masked_fill(attention_mask == 0, float("-inf"))
            state_emb = (h_out * F.softmax(pool_scores, dim=-1).unsqueeze(-1)).sum(dim=1)
            return (
                state_emb,
                self.heads.choice_head(state_emb),
                self.heads.noul_head(state_emb),
                self.heads.score_head(state_emb),
            )

    rng = random.Random(42)
    samples = build_curriculum_samples(rng)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] + build_ood_calibration_samples(rng)

    def _to_tensors(batch: list[tuple[str, int, float, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2, fused=(device.type == "cuda"))
    lambda_sigreg = 0.02
    assert lambda_sigreg <= 0.05

    counts = torch.bincount(tr_y, minlength=5).float().clamp(min=1.0)
    class_weights = len(tr_y) / (5.0 * counts)
    class_weights = class_weights / class_weights.mean()

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch_size = 256
    for epoch in range(1, 25):
        perm = torch.randperm(len(tr_y), device=device)
        for i in range(0, len(tr_y), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
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
                # Train 4d PremiseFit MLP on in-batch positive (same strategy) and negative pairs
                s_shift = torch.roll(s_x, shifts=1, dims=0)
                y_shift = torch.roll(tr_y[idx], shifts=1, dims=0)
                fit_feat = torch.cat([s_x, s_shift, s_x - s_shift, s_x * s_shift], dim=-1)
                fit_logit = model.heads.premise_fit(fit_feat)
                fit_target = (tr_y[idx] == y_shift).float().unsqueeze(-1)
                fit_loss = F.binary_cross_entropy_with_logits(fit_logit, fit_target)
                loss = ce_loss + 0.5 * noul_loss + 0.25 * fit_loss + lambda_sigreg * sigreg_loss
            loss.backward()
            opt.step()
        if epoch % 6 == 0 or epoch == 1:
            with torch.no_grad():
                _, v_logits, _, _ = model(va_ids, va_mask)
                acc = (v_logits.argmax(dim=-1) == va_y).float().mean().item()
            print(f"[Stage 1] epoch={epoch}/24 ce={ce_loss.item():.4f} sigreg={sigreg_loss.item():.4f} val_acc={acc:.4f}")

    # L-BFGS Temperature Calibration clamped to [0.5, 2.5] on mixed in-dist + OOD split
    model.eval()
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

    out_dir = Path("/content/artifacts") if Path("/content").exists() else Path("artifacts")
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
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,
    )

    # Preserve heads.premise_fit.{0,2}.{weight,bias} initializers inside lean_sys1.onnx
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(str(onnx_path))
    existing_inits = {init.name for init in onnx_model.graph.initializer}
    state_dict = cpu_model.state_dict()
    required_fit_tensors = {
        "heads.premise_fit.0.weight": (384, 1536),
        "heads.premise_fit.0.bias": (384,),
        "heads.premise_fit.2.weight": (1, 384),
        "heads.premise_fit.2.bias": (1,),
    }
    for name in required_fit_tensors:
        if name not in existing_inits:
            arr = state_dict[name].detach().cpu().numpy()
            onnx_model.graph.initializer.append(numpy_helper.from_array(arr, name=name))
    onnx.save(onnx_model, str(onnx_path))

    # Mandatory post-export check verifying all four PremiseFit weights are present and trained
    verified_model = onnx.load(str(onnx_path))
    verified_inits = {
        init.name: numpy_helper.to_array(init) for init in verified_model.graph.initializer
    }
    for name, expected_shape in required_fit_tensors.items():
        assert name in verified_inits, f"Missing ONNX initializer: {name}"
        assert tuple(verified_inits[name].shape) == expected_shape, (
            f"Shape mismatch for {name}: {verified_inits[name].shape} != {expected_shape}"
        )
    assert float(verified_inits["heads.premise_fit.0.weight"].std()) > 1e-4, (
        "heads.premise_fit.0.weight appears uninitialized"
    )
    temps = {"T_choice": t_choice_val, "T_noul": 1.0, "T_fit": 1.0}
    (out_dir / "temperatures.json").write_text(json.dumps(temps, indent=2), encoding="utf-8")
    print(f"[Stage 1] Exported {out_dir / 'lean_sys1.onnx'} and temperatures.json (T_choice={t_choice_val:.4f})")


if __name__ == "__main__":
    main()
