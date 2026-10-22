#!/usr/bin/env python3
"""G0 — restore training wall-clock instrumentation in four trainers.

DEFECT (found 2026-10-04): train_ablation_a5/a6/a7.py and train_track_b.py open
`mlflow.start_run()` AFTER `model.learn()` returns.  The MLflow run therefore
covers only the logging calls, so `end_time - start_time` is ~0 AND no
`wall_time_min` metric is written at all.  Historical A5/A6/A7/TrackB runs have
no recoverable duration.  `train_rl_inv.py` is correct and is the reference:
it both wraps `learn()` in the run and logs an explicit metric.

FIX — all three parts, matching the reference pattern:
  1. MOVE `mlflow.start_run()` so the run ENCOMPASSES `model.learn()`
     (makes MLflow's own end_time - start_time meaningful)
  2. time `learn()` explicitly
  3. log `wall_time_min`, as train_rl_inv.py:313 already does

SAFE BY CONSTRUCTION: these four trainers pass `callback=eval_cb` only -- none
uses MlflowRolloutCallback -- so nothing logs to MLflow during training and
moving the run boundary changes no logging behaviour.  No environment, RNG,
hyperparameter or checkpoint path is touched, so training dynamics and all
existing results are unaffected.

FAIL-CLOSED: every file is patched in memory and ALL THREE parts verified
before anything is written.  If any expected pattern is missing in any file,
the script aborts without modifying a single file.

    python tools/fix_mlflow_timing.py --repo . --check    # preview, writes nothing
    python tools/fix_mlflow_timing.py --repo .            # apply
"""
from __future__ import annotations
import argparse, ast, re, shutil, sys
from pathlib import Path

TARGETS = {
    "src/train/train_ablation_a5.py": "A5",
    "src/train/train_ablation_a6.py": "A6",
    "src/train/train_ablation_a7.py": "A7",
    "src/train/train_track_b.py":     "TrackB",
}
LEARN_RE = re.compile(r"^(\s*)model\.learn\(total_timesteps=args\.timesteps, callback=eval_cb\)\s*$")
WITH_RE  = re.compile(r"^(\s*)with mlflow\.start_run\((.*)\):\s*$")


def patch(text: str, label: str):
    """Return (new_text, notes, ok). Performs all three parts or fails."""
    notes = []
    lines = text.splitlines()

    # locate model.learn(...) and the LATER `with mlflow.start_run(...):`
    li = next((i for i, l in enumerate(lines) if LEARN_RE.match(l)), None)
    if li is None:
        return text, ["model.learn(...) not found in the expected form"], False
    learn_indent = LEARN_RE.match(lines[li]).group(1)

    wi = next((i for i in range(li + 1, len(lines)) if WITH_RE.match(lines[i])), None)
    if wi is None:
        return text, ["`with mlflow.start_run(...):` not found AFTER model.learn()"], False
    wm = WITH_RE.match(lines[wi])
    with_indent, run_args = wm.group(1), wm.group(2)
    if with_indent != learn_indent:
        return text, [f"indent mismatch: learn={len(learn_indent)} with={len(with_indent)}"], False

    # end of the with-body: first later non-blank line indented <= the with line
    base = len(with_indent)
    end = len(lines)
    for i in range(wi + 1, len(lines)):
        if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= base:
            end = i
            break

    region_a = lines[li:wi]          # learn(...) .. just before the old `with`
    body     = lines[wi + 1:end]     # the existing with-body (already +4)

    # MLflow SESSION setup must stay OUTSIDE the run: train_track_b.py calls
    # set_experiment() between learn() and the old start_run(), so naively
    # indenting it into the new run creates the run in the wrong experiment.
    # Hoist those lines above the new `with`.
    setup_re = re.compile(r"^\s*mlflow\.(set_tracking_uri|set_experiment)\(")
    hoisted = [l.strip() for l in region_a if setup_re.match(l)]
    region_a = [l for l in region_a if not setup_re.match(l)]
    hoisted_lines = [f"{with_indent}{h}" for h in hoisted]
    if hoisted_lines:
        notes.append(f"hoisted {len(hoisted_lines)} mlflow session-setup line(s) above the run")

    timed = [
        f"{learn_indent}    _t_start = datetime.now()",
        f"{learn_indent}    model.learn(total_timesteps=args.timesteps, callback=eval_cb)",
        f"{learn_indent}    wall_time_min = (datetime.now() - _t_start).total_seconds() / 60.0",
        f'{learn_indent}    print(f"[{label}] training wall-clock: {{wall_time_min:.1f}} min")',
    ]
    rest_a = [("    " + l) if l.strip() else l for l in region_a[1:]]

    new = (lines[:li]
           + hoisted_lines
           + [f"{with_indent}with mlflow.start_run({run_args}):"]   # run now WRAPS learn()
           + timed + rest_a + body + lines[end:])
    text = "\n".join(new) + ("\n" if text.endswith("\n") else "")
    notes.append("moved start_run() to wrap learn()")
    notes.append("timed learn()")

    # add the metric to the existing log_metrics call
    before = text
    text = re.sub(r'(mlflow\.log_metrics\(\{\s*\n\s*"best_eval_reward":[^\n]*\n)',
                  r'\1            "wall_time_min":    round(wall_time_min, 1),\n',
                  text, count=1)
    if text == before:
        return text, notes + ["could not add wall_time_min to log_metrics"], False
    notes.append("logged wall_time_min")

    if "from datetime import datetime" not in text:
        text = re.sub(r"^(import os\s*$)", r"from datetime import datetime\n\1",
                      text, count=1, flags=re.M)
        notes.append("added datetime import")
    return text, notes, True


def verify(text: str) -> list[str]:
    """All three parts present, and the file parses."""
    problems = []
    try:
        ast.parse(text)
    except SyntaxError as e:
        problems.append(f"syntax error: {e}")
        return problems
    li = next((i for i, l in enumerate(text.splitlines()) if LEARN_RE.match(l)), None)
    wi = next((i for i, l in enumerate(text.splitlines()) if WITH_RE.match(l)), None)
    if li is None or wi is None:
        problems.append("learn() or start_run() missing after patch")
    elif wi > li:
        problems.append("start_run() is STILL after learn() -- the move did not apply")
    if "_t_start = datetime.now()" not in text:
        problems.append("learn() is not timed")
    if '"wall_time_min"' not in text:
        problems.append("wall_time_min not logged")
    # set_experiment() must EXECUTE before start_run().  Compare by enclosing
    # function, not by line number: a6 calls set_experiment() inside main(),
    # which is defined later in the file but runs earlier.  Only a set_experiment
    # that is in the SAME function and textually after start_run is a defect.
    L = text.splitlines()
    sr = next((i for i, l in enumerate(L) if WITH_RE.match(l)), None)
    if sr is not None:
        tree = ast.parse(text)
        def enclosing(line_no):
            best = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.lineno <= line_no <= (node.end_lineno or node.lineno):
                        if best is None or node.lineno > best.lineno:
                            best = node
            return best
        fn = enclosing(sr + 1)
        if fn is not None:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "set_experiment"
                        and node.lineno > sr + 1):
                    problems.append(
                        f"mlflow.set_experiment() at line {node.lineno} is inside the "
                        f"same function and AFTER start_run() -- the run would be "
                        f"created in the wrong experiment")
                    break
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    repo = a.repo.expanduser().resolve()

    # ---- PHASE 1: patch everything in memory, verify, abort on any failure ----
    staged, failed = {}, False
    for rel, label in TARGETS.items():
        p = repo / rel
        if not p.exists():
            print(f"  {rel:38s} MISSING"); failed = True; continue
        src = p.read_text(encoding="utf-8")
        if '"wall_time_min"' in src and src.index("with mlflow.start_run") < src.index("model.learn"):
            print(f"  {rel:38s} already fully patched"); continue
        out, notes, ok = patch(src, label)
        problems = verify(out) if ok else notes
        if not ok or problems:
            print(f"  {rel:38s} FAILED  [{'; '.join(problems)}]"); failed = True; continue
        staged[p] = out
        print(f"  {rel:38s} {'WOULD PATCH' if a.check else 'ready'}  [{'; '.join(notes)}]")

    if failed:
        print("\n  ABORTED -- no file was modified. Fix the mismatch and re-run.")
        sys.exit(1)
    if a.check:
        print(f"\n  --check: {len(staged)} file(s) would be patched, nothing written.")
        sys.exit(0)

    # ---- PHASE 2: only now write ----
    for p, out in staged.items():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(out, encoding="utf-8")
    print(f"\n  wrote {len(staged)} file(s); backups as *.py.bak")


if __name__ == "__main__":
    main()
