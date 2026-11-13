#!/usr/bin/env python
"""Krok 6: ED z angle_seed 1,2,3 (tagi _as1.._as3) wobec ED_seed0 i 5 powtorzen ED_seed0_nf0..4. Tylko odczyt eval.json.
Wynik: step6_summary.md (fail-if-exists)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; EV = REPO / "outputs/ml/eval"; OUT = Path(__file__).resolve().parent / "step6_summary.md"
if OUT.exists(): sys.exit("step6_summary.md juz istnieje")
def r(i):
    p = EV / i / "eval.json"
    if not p.exists(): return None
    d = json.loads(p.read_text(encoding="utf-8"))["test_sets"]["test@36"]
    return d["overall"]["all"]["RMSE"], {k: v["all"]["RMSE"] for k, v in d["per_scene"].items()}
nf = [r(f"ED_seed0_nf{i}") for i in range(5)]; asr = [r(f"ED_seed0_as{i}") for i in (1, 2, 3)]; o = r("ED_seed0")
L = ["# Krok 6: ED z różnym `angle_seed` (training seed 0, 40 000 kroków, AMP)", "",
     "| przebieg | angle_seed | RMSE test@36 |", "|---|---|---|", f"| ED_seed0 (oryginał, sierpień) | 0 | {o[0]:.5f} |"]
for i, x in enumerate(nf): L.append(f"| ED_seed0_nf{i} (powtórzenie) | 0 | {x[0]:.5f} |" if x else f"| ED_seed0_nf{i} | 0 | brak |")
for i, x in zip((1, 2, 3), asr): L.append(f"| ED_seed0_as{i} | {i} | {x[0]:.5f} |" if x else f"| ED_seed0_as{i} | {i} | brak |")
v_nf = np.array([x[0] for x in nf if x]); v_as = np.array([x[0] for x in asr if x])
L += ["", f"- powtórzenia `nf` (angle_seed 0): średnia {v_nf.mean():.5f}, sd {v_nf.std(ddof=1):.5f}, rozstęp {np.ptp(v_nf):.5f}."]
if v_as.size:
    L.append(f"- `as1..3` (n = {v_as.size}): średnia {v_as.mean():.5f}, sd {v_as.std(ddof=1) if v_as.size > 1 else float('nan'):.5f}, rozstęp {np.ptp(v_as):.5f}; "
             f"średnia `as` − średnia `nf` = {v_as.mean() - v_nf.mean():+.5f}; oryginał − średnia `as` = {o[0] - v_as.mean():+.5f}.")
    inr = [(f'as{i}', bool(v_nf.min() <= x[0] <= v_nf.max())) for i, x in zip((1, 2, 3), asr) if x]
    L.append(f"- czy wynik mieści się w zakresie powtórzeń `nf`: {inr}")
L += ["", "Interpretacja (hipotezy, nie wnioski): jeśli `as1..3` mają rozrzut podobny do `nf`, to losowanie 4 orientacji na lokalizację (angle_seed) "
      "nie wnosi wykrywalnej dodatkowej zmienności ponad niedeterminizm GPU; jeśli rozstęp `as` jest wyraźnie większy — angle_seed jest osobnym źródłem rozrzutu."]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L[-6:]))
