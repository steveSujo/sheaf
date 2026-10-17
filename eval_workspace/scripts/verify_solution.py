"""End-to-end 4-gate formal verification runner for `eval_workspace/`."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ALLOWED_AXIOMS: frozenset[str] = frozenset({"propext", "Classical.choice", "Quot.sound"})


def extract_signature(lean_src: str, thm_short_name: str) -> str:
    """Extract whitespace-normalized theorem signature before `:= by`."""
    pattern = rf"(theorem\s+{re.escape(thm_short_name)}\b.*?):=\s*by"
    m = re.search(pattern, lean_src, re.DOTALL)
    if not m:
        raise ValueError(f"Theorem signature for '{thm_short_name}' not found")
    return " ".join(m.group(1).split())


def verify(workspace: Path, entry: dict[str, Any]) -> tuple[bool, str]:
    """Run all 4 verification gates on `solutions/<id>.lean`."""
    prob_path = workspace / str(entry["file"])
    sol_path = workspace / "solutions" / f"{entry['id']}.lean"
    if not sol_path.exists():
        return False, "Solution file missing"

    short_name = str(entry["theorem"]).split(".")[-1]
    prob_sig = extract_signature(prob_path.read_text(encoding="utf-8"), short_name)
    sol_src = sol_path.read_text(encoding="utf-8")
    sol_sig = extract_signature(sol_src, short_name)

    # Gate 1: Signature Lock
    if prob_sig != sol_sig:
        return False, "Signature mismatch: theorem statement was modified"

    # Gate 2 & 3: Kernel Elaboration + Axiom Audit
    audit_src = f"{sol_src}\n\n#print axioms {entry['theorem']}\n"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lean", dir=workspace, delete=False, encoding="utf-8"
    ) as tf:
        tf.write(audit_src)
        tmp_path = Path(tf.name)

    try:
        res = subprocess.run(
            ["lake", "env", "lean", str(tmp_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=int(entry["timeout_s"]),
            check=False,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if res.returncode != 0:
        return False, f"Kernel error:\n{res.stderr or res.stdout}"

    if "declaration uses 'sorry'" in res.stdout or "sorryAx" in res.stdout:
        return False, "Illegal axioms used: ['sorryAx']"

    m = re.search(r"depends on axioms:\s*\[([^\]]*)\]", res.stdout)
    if m:
        used = {a.strip() for a in m.group(1).split(",") if a.strip()}
        illegal = used - ALLOWED_AXIOMS
        if illegal:
            return False, f"Illegal axioms used: {sorted(illegal)}"

    return True, "VERIFIED"


if __name__ == "__main__":
    ws = Path(sys.argv[1])
    target_id = sys.argv[2]
    for line in (ws / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["id"] == target_id:
            ok, msg = verify(ws, entry)
            print(f"[{target_id}] {msg}")
            sys.exit(0 if ok else 1)
    sys.exit(2)
