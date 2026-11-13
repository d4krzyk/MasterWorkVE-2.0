#!/usr/bin/env python
"""Podloga szumu niedeterminizmu GPU na PELNYM budzecie (40 000 krokow).

    python my-operations/ml/checks/noise_floor.py run [--n 5] [--condition ED] [--seed 0]
    python my-operations/ml/checks/noise_floor.py summarize
    python my-operations/ml/checks/noise_floor.py run --n 1 --seeds 1 2 \
        --out noise_floor_ED_seeds12_40k.json      # kilka ziaren w jednym pliku

Ten sam warunek, to samo ziarno treningu, identyczna konfiguracja co oryginalny `ED_seed0`
(AMP wlaczone, bez `use_deterministic_algorithms`, bez `--fast-bilinear`), N powtorzen. Jedyne
zrodlo rozrzutu to niedeterminizm jader GPU/cuDNN i kolejnosc operacji zmiennoprzecinkowych.

Przebiegi ida do `outputs/ml/runs/<warunek>_seed<N>_nf<i>` i `outputs/ml/eval/<...>_nf<i>`, wiec
nie dotykaja `ED_seed0`. Po kazdym treningu leci `evaluate.py --test-set test@36`.
`run` jest WZNAWIALNE: przebieg z kompletnym `status.json` (finished) i `eval.json` jest pomijany,
przerwany trening idzie od zera (`--force`) -- polowiczny przebieg nie wchodzi do statystyki.
`summarize` nie dotyka GPU i zapisuje `outputs/ml/determinism/noise_floor_<warunek>_40k.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.checks"

from .. import paths  # noqa: E402

TRAIN = paths.MY_OPERATIONS / "ml" / "depth_model" / "train_condition.py"
EVAL = paths.MY_OPERATIONS / "ml" / "depth_model" / "evaluate.py"
OUT_DIR = paths.ML_OUTPUTS / "determinism"


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _run_id(cond: str, seed: int, i: int) -> str:
    return f"{cond}_seed{seed}_nf{i}"


def _finished(run_id: str) -> bool:
    st = _load(paths.RUNS_DIR / run_id / "status.json")
    return bool(st and st.get("finished") and not st.get("interrupted"))


def _rmse36(run_id: str) -> float | None:
    d = _load(paths.ML_OUTPUTS / "eval" / run_id / "eval.json")
    try:
        return float(d["test_sets"]["test@36"]["overall"]["all"]["RMSE"])
    except (KeyError, TypeError):
        return None


def cmd_run(args) -> int:
    log_dir = OUT_DIR / f"noise_floor_{args.condition}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rc = _run_one_seed(args, seed, log_dir)
        if rc:
            return rc
    return cmd_summarize(args)


def _run_one_seed(args, seed: int, log_dir: Path) -> int:
    for i in range(args.n):
        rid = _run_id(args.condition, seed, i)
        if _finished(rid) and _rmse36(rid) is not None:
            print(f"[{i + 1}/{args.n}] {rid}: gotowy, pomijam")
            continue
        run_dir = paths.RUNS_DIR / rid
        if not _finished(rid):
            cmd = [sys.executable, str(TRAIN), "--condition", args.condition,
                   "--seed", str(seed), "--tag", f"nf{i}"]
            if run_dir.exists():
                cmd.append("--force")           # przerwany przebieg: od zera
            print(f"[{i + 1}/{args.n}] trening: {' '.join(cmd)}", flush=True)
            with (log_dir / f"{rid}_train.log").open("wb") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(paths.REPO_ROOT)).returncode
            if rc != 0 or not _finished(rid):
                print(f"BLAD: trening {rid} rc={rc}; zatrzymuje (log: {log_dir})")
                return 1
        cmd = [sys.executable, str(EVAL), "--run-dir", str(run_dir), "--test-set", "test@36"]
        print(f"[{i + 1}/{args.n}] ewaluacja: {rid}", flush=True)
        with (log_dir / f"{rid}_eval.log").open("wb") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=str(paths.REPO_ROOT)).returncode
        if rc != 0 or _rmse36(rid) is None:
            print(f"BLAD: ewaluacja {rid} rc={rc}")
            return 1
        print(f"    {rid}: RMSE test@36 = {_rmse36(rid):.5f}", flush=True)
        cmd_summarize(args, quiet=True)           # wynik czastkowy po kazdym przebiegu
    return 0


def _summary_one_seed(args, seed: int) -> dict:
    runs = []
    for i in range(args.n):
        rid = _run_id(args.condition, seed, i)
        r = _rmse36(rid)
        st = _load(paths.RUNS_DIR / rid / "status.json") or {}
        if r is not None and _finished(rid):
            runs.append({"run_id": rid, "RMSE_test36": r, "best_step": st.get("best_step"),
                         "best_val36_rmse": st.get("best_val36_rmse")})
    v = np.array([r["RMSE_test36"] for r in runs])
    ref_id = f"{args.condition}_seed{seed}"
    ref = _rmse36(ref_id)
    ref_st = _load(paths.RUNS_DIR / ref_id / "status.json") or {}
    out = {
        "opis": "Podloga szumu niedeterminizmu GPU na pelnym budzecie: ten sam warunek i ziarno "
                "treningu, identyczna konfiguracja co oryginal (AMP, bez "
                "use_deterministic_algorithms), tylko niedeterminizm GPU rozni przebiegi.",
        "warunek": args.condition, "ziarno": seed, "kroki": 40000, "metryka": "RMSE test@36 (all)",
        "n_przebiegow": len(runs), "przebiegi": runs,
        "srednia": float(v.mean()) if v.size else None,
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "rozstep": float(v.max() - v.min()) if v.size else None,
        "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
        "oryginal": {"run_id": ref_id, "RMSE_test36": ref, "best_step": ref_st.get("best_step"),
                     "best_val36_rmse": ref_st.get("best_val36_rmse")},
    }
    if v.size and ref is not None:
        out["oryginal"].update({
            "roznica_od_sredniej_powtorzen": float(ref - v.mean()),
            "w_zakresie_powtorzen": bool(v.min() <= ref <= v.max()),
            "z_wzgledem_sd": (float((ref - v.mean()) / v.std(ddof=1)) if v.size > 1 else None)})
    return out


def cmd_summarize(args, quiet: bool = False) -> int:
    # Jedno ziarno: format pliku DOKLADNIE jak dotad. Kilka ziaren: blok per ziarno.
    per_seed = {str(s): _summary_one_seed(args, s) for s in args.seeds}
    out = (next(iter(per_seed.values())) if len(per_seed) == 1 else
           {"opis": "Powtorzenia przy kilku ziarnach treningu; blok per ziarno ma ten sam "
                    "format co plik jednego ziarna.",
            "warunek": args.condition, "ziarna": list(args.seeds), "per_ziarno": per_seed})
    path = OUT_DIR / (args.out or f"noise_floor_{args.condition}_40k.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if not quiet:
        for s, o in per_seed.items():
            print(f"ziarno {s}:", json.dumps({k: o[k] for k in ("n_przebiegow", "srednia", "sd",
                                                                 "rozstep", "oryginal")},
                                             indent=2, ensure_ascii=False))
        print(f"zapisano: {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--condition", default="ED")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="kilka ziaren po kolei (zastepuje --seed); domyslnie [--seed]")
    ap.add_argument("--out", default=None,
                    help="nazwa pliku w outputs/ml/determinism/; domyslnie "
                         "noise_floor_<warunek>_40k.json (jak dotad)")
    args = ap.parse_args(argv)
    args.seeds = args.seeds or [args.seed]
    if args.out and ("/" in args.out or not args.out.endswith(".json")):
        ap.error("--out to sama nazwa pliku .json (bez katalogu)")
    return cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
