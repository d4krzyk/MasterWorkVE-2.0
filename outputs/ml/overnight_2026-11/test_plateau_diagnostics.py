#!/usr/bin/env python
"""Test poprawki _plateau_diagnostics (klucz "all"): syntetyczny metrics.jsonl + porownanie z
plateau_recomputed.md dla ED_seed0, B_seed0, EA_seed0. Tylko odczyt runs/. Uruchom z korzenia repo."""
import json, re, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "my-operations"))
from ml.depth_model.train_condition import _plateau_diagnostics as pd_  # noqa: E402

def rows(vals, key="all"):
    return "\n".join(json.dumps({"step": 1000 * (i + 1), "split": "val", key: {"RMSE": v}}) for i, v in enumerate(vals)) + "\n"

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    # syntetyk: 40 punktow; kwartaly 3 i 4: prev = 1.0 (stale), last = 0.9 .. 0.81 (10 pkt)
    vals = [2.0] * 20 + [1.0] * 10 + [0.9 - 0.01 * k for k in range(10)]
    (t / "a.jsonl").write_text(rows(vals))
    r = pd_(t / "a.jsonl", 40000)
    last = np.array(vals[30:])
    assert r["n_validations"] == 40 and r["plateau_from_step"] == 30000, r
    assert abs(r["plateau_sd"] - last.std()) < 1e-12, r
    assert abs(r["improvement_last_quarter"] - (1.0 - last.min())) < 1e-12, r
    # stary klucz "overall" NIE jest czytany (stare wiersze 80-wierszowych plikow sa pomijane)
    (t / "b.jsonl").write_text(rows(vals, "overall"))
    assert pd_(t / "b.jsonl", 40000)["n_validations"] == 0
    # mieszany plik 40 starych + 40 nowych: liczy sie tylko nowe
    (t / "c.jsonl").write_text(rows([9.0] * 40, "overall") + rows(vals))
    rc = pd_(t / "c.jsonl", 40000)
    assert rc["n_validations"] == 40 and abs(rc["plateau_sd"] - last.std()) < 1e-12, rc
print("syntetyczne: OK")

md = (REPO / "outputs/ml/overnight_2026-11/plateau_recomputed.md").read_text(encoding="utf-8")
for rid in ("ED_seed0", "B_seed0", "EA_seed0"):
    line = next(l for l in md.splitlines() if l.startswith(f"| {rid} |"))
    c = [x.strip() for x in line.strip("|").split("|")]
    sd_q = float(c[5])
    L = [json.loads(l) for l in (REPO / f"outputs/ml/runs/{rid}/metrics.jsonl").read_text().splitlines()][-40:]
    v = [x["all"]["RMSE"] for x in L]
    d = pd_(REPO / f"outputs/ml/runs/{rid}/metrics.jsonl", 40000)
    sd_exp = sd_q * np.sqrt(9 / 10)          # md: ddof=1, funkcja: np.std (ddof=0)
    imp_exp = float(np.median(v[20:30]) - min(v[30:]))
    print(f"{rid}: n={d['n_validations']} sd fun={d['plateau_sd']:.6f} md*sqrt(.9)={sd_exp:.6f} | "
          f"popr fun={d['improvement_last_quarter']:.6f} wlasne={imp_exp:.6f} | md sd_q={c[5]}")
    assert d["n_validations"] == 40 and abs(d["plateau_sd"] - sd_exp) < 1e-5
    assert abs(d["improvement_last_quarter"] - imp_exp) < 1e-9
print("zgodnosc z plateau_recomputed.md: OK")
