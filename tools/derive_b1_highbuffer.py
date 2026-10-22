#!/usr/bin/env python3
"""G3 supplemental — derive B1*-HighBuffer and complete the provenance record.

Reads the FROZEN calibration record and derives a fourth, exploratory variant
from the candidate evaluations that already exist.  **No episodes are re-run**
and the frozen B1*-Deployable selection is not touched.

G3 exploratory amendment, 22 October 2026 — buffer-priority tie-break
---------------------------------------------------------------------
Rule: among candidates at the minimum observed training-window EENS (tolerance
1e-12), select the HIGHEST reorder threshold s.  Remaining ties resolve by
lower mean diesel, then fewer mean orders, then deterministic enumeration order.

Introduced AFTER inspecting calibration results but BEFORE any held-out
evaluation.  Reported as a post-calibration, pre-test exploratory sensitivity
analysis -- NOT as a preregistered baseline.  Both B1*-Deployable and
B1*-HighBuffer are reported for every site and scenario regardless of which
performs better.

Also SUPPLEMENTS (does not close) ECG-1 by recording SHA-256 of all ten
processed site CSVs, which the frozen calibration record omitted.  These are
single-platform raw-byte digests; CRLF/LF conversion changes the digest without
changing the data, so ECG-1 closure still requires row-wise numerical
comparison after newline normalisation (protocol §1.9).

    python tools/derive_b1_highbuffer.py --repo .
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

TOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_csv(data_dir: Path, i: int) -> Path:
    """All ten processed site CSVs are required; silently skipping one would
    leave a provenance record that looks complete and is not."""
    p = data_dir / f"site{i}.csv"
    if not p.exists():
        raise SystemExit(f"required dataset file missing: {p}")
    return p


def _platform_tag() -> str:
    import platform
    return f"{platform.system()} {platform.release()} / python {platform.python_version()}"


def pick_highbuffer(candidates):
    best = min(c["eens_mean"] for c in candidates)
    tied = [c for c in candidates if abs(c["eens_mean"] - best) <= TOL]
    s_max = max(c["s"] for c in tied)
    at_s = [c for c in tied if abs(c["s"] - s_max) <= TOL]
    chosen = sorted(at_s, key=lambda c: (c["diesel_mean"], c["orders_mean"]))[0]
    return chosen, len(tied), len(at_s)


def identical(a, b, key, tol):
    return all(abs(x - y) <= tol for x, y in zip(a[key], b[key]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--calibration", type=Path,
                    default=Path("results/phase3r/b1_star_calibration.json"))
    ap.add_argument("--params-in", type=Path,
                    default=Path("results/phase3r/b1_star_params.json"),
                    help="frozen calibration artefact -- READ ONLY, never written")
    ap.add_argument("--params-out", type=Path,
                    default=Path("results/phase3r/b1_variants_params.json"),
                    help="new four-variant file consumed by the evaluator")
    ap.add_argument("--out", type=Path,
                    default=Path("results/phase3r/b1_highbuffer_amendment.json"))
    a = ap.parse_args()
    repo = a.repo.expanduser().resolve()
    calib_p = a.calibration if a.calibration.is_absolute() else repo / a.calibration
    params_in = a.params_in if a.params_in.is_absolute() else repo / a.params_in
    params_out = a.params_out if a.params_out.is_absolute() else repo / a.params_out

    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    frozen_sha_before = sha256(params_in)
    params = json.loads(params_in.read_text(encoding="utf-8"))

    rows, hb_params = [], {}
    for site in calib["sites"]:
        w = site["selected_stats"]
        hb, n_tied, n_at_s = pick_highbuffer(site["all_candidates"])
        hb_params[site["site"]] = {"s_norm": hb["s"], "S_norm": hb["S"]}
        rows.append({
            "site": site["site"],
            "B1_star_Deployable": {"s_norm": w["s"], "S_norm": w["S"],
                                   "orders_mean": w["orders_mean"]},
            "B1_star_HighBuffer": {"s_norm": hb["s"], "S_norm": hb["S"],
                                   "orders_mean": hb["orders_mean"]},
            "candidates_tied_at_min_EENS": n_tied,
            "candidates_at_max_s": n_at_s,
            "same_candidate": (abs(w["s"] - hb["s"]) <= TOL
                               and abs(w["S"] - hb["S"]) <= TOL),
            "calibration_episode_identity": {
                "eens_identical":   identical(w, hb, "eens_episodes", TOL),
                "diesel_identical": identical(w, hb, "diesel_episodes", 1e-9),
                "orders_identical": identical(w, hb, "orders_episodes", 1e-9),
            },
        })

    params["B1_star_HighBuffer"] = hb_params
    params_out.parent.mkdir(parents=True, exist_ok=True)
    params_out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    # The frozen artefact must be byte-identical after this script runs.
    if sha256(params_in) != frozen_sha_before:
        raise AssertionError("frozen b1_star_params.json was modified -- aborting")

    data_dir = repo / "data" / "processed"
    report = {
        "status": "OK",
        "amendment": "G3 exploratory amendment — buffer-priority tie-break",
        "amendment_date": "2026-10-22",
        "standing": ("post-calibration EXPLORATORY sensitivity analysis. NOT a "
                     "preregistered baseline."),
        "test_exposure_disclosure": (
            "HighBuffer's selection rule was specified from training-window "
            "findings BEFORE any held-out evaluation. The full evaluation "
            "protocol and the decision to report all four variants regardless of "
            "outcome were fixed BEFORE the full-grid run. However, a limited "
            "held-out pilot (site5, normal+delayed, seeds 42/123) was executed "
            "and inspected before the full grid. No parameter was selected using "
            "it, but the full-grid results must NOT be described as a completely "
            "untouched test set for HighBuffer."),
        "rule": ("among candidates at minimum training EENS (tol 1e-12): highest s; "
                 "then lower diesel; then fewer orders; then enumeration order"),
        "hypothesis": ("earlier replenishment MAY improve reliability under longer "
                       "delivery lead times, at the cost of more frequent ordering. "
                       "Higher s does NOT guarantee higher realised inventory: "
                       "deliveries are stochastic and order quantities are fixed."),
        "scope_limit": ("HighBuffer is the maximum-s candidate within the ALREADY "
                        "EVALUATED grid. The refinement pass was centred on the "
                        "B1*-Deployable winner, so this is not a separately "
                        "optimised global maximum-buffer solution."),
        "reporting_rule": ("B1*-Deployable and B1*-HighBuffer are both reported for "
                           "every site and scenario, regardless of which performs "
                           "better. HighBuffer is NOT propagated to MPC or "
                           "Oracle-MPC without a further recorded scope amendment."),
        "episodes_rerun": 0,
        "frozen_calibration_untouched": True,
        "source_calibration": {"path": str(calib_p.relative_to(repo)),
                               "sha256": sha256(calib_p)},
        "sites": rows,
        "ECG1_supplemental_dataset_hashes": {
            "status": "SUPPLEMENTS ECG-1 -- DOES NOT CLOSE IT",
            "note": ("The frozen calibration record omitted these hashes; they are "
                     "recorded here rather than re-running 5,030 episodes. These "
                     "are single-platform raw-byte digests. Because CRLF/LF "
                     "conversion changes the digest without changing the data "
                     "(protocol §1.9), hashes ALONE cannot establish cross-platform "
                     "dataset identity. ECG-1 closure still requires row-wise "
                     "numerical comparison after newline normalisation."),
            "platform": _platform_tag(),
            "files": {f"data/processed/site{i}.csv": sha256(_require_csv(data_dir, i))
                      for i in range(1, 11)},
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = a.out if a.out.is_absolute() else repo / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("B1*-HighBuffer derived from frozen calibration — 0 episodes re-run\n")
    print("%-8s %-14s %-14s %9s %9s  %s" % ("site", "B1*", "HighBuffer",
                                            "ord B1*", "ord HB", "EENS/diesel identical"))
    for r in rows:
        d, h = r["B1_star_Deployable"], r["B1_star_HighBuffer"]
        ident = r["calibration_episode_identity"]
        print("%-8s (%.2f, %.2f)   (%.2f, %.2f)   %9.2f %9.2f  %s / %s" % (
            r["site"], d["s_norm"], d["S_norm"], h["s_norm"], h["S_norm"],
            d["orders_mean"], h["orders_mean"],
            ident["eens_identical"], ident["diesel_identical"]))
    n_same = sum(r["same_candidate"] for r in rows)
    print(f"\n  sites where HighBuffer == B1*: {n_same}/10")
    print(f"  frozen artefact: {params_in} (unchanged, sha {frozen_sha_before[:12]})")
    print(f"  four-variant   : {params_out}")
    print(f"  amendment      : {out}")


if __name__ == "__main__":
    main()
