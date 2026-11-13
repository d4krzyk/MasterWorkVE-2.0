#!/usr/bin/env python
"""Krok 7: kontrola RGB-only (sama RGBDepthNet od zera) na tych samych parach co A, D, B, wobec pelnego modelu.
Tylko odczyt eval.json. Wynik: step7_summary.md (fail-if-exists). Konwencja: X − Y dosłownie; test t po ziarnach = ttest_1samp."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
REPO = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent; OUT = HERE / "step7_summary.md"
if OUT.exists(): sys.exit("step7_summary.md juz istnieje")
S = (0, 1, 2)
def ev(p, ts):
    p = Path(p)
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))["test_sets"][ts]["overall"]["all"]["RMSE"]
def col(kind, c, ts):
    base = (HERE / "rgbonly" / f"RGBONLY_{c}_seed{{}}" / "eval.json") if kind == "rgb" else (REPO / f"outputs/ml/eval/{c}_seed{{}}/eval.json")
    v = [ev(str(base).format(s), ts) for s in S]
    return np.array(v, dtype=float) if all(x is not None for x in v) else None
def ms(v): return "brak" if v is None else f"{v.mean():.5f} ± {v.std(ddof=1):.5f}"
def pt(a, b):
    if a is None or b is None: return "brak"
    d = a - b; t = stats.ttest_1samp(d, 0.0)
    return f"{d.mean():+.5f} ± {d.std(ddof=1):.5f} (p = {t.pvalue:.4f})"
L = ["# Krok 7: kontrola RGB-only (RGB2Depth od zera) wobec modelu pełnego", "",
     "RGB-only = sama `RGBDepthNet`, te same pary (lokalizacja, kąt) co A/D/B, 40 000 kroków, batch 32, wybór checkpointu po walidacji na 36 orientacjach. "
     "Pełny model = A/D/B z `outputs/ml/eval`. RMSE głębi [m], 3 ziarna.", ""]
for ts in ("test@36", "test@4"):
    R = {c: col("rgb", c, ts) for c in "ADB"}; F = {c: col("full", c, ts) for c in "ADB"}
    L += [f"## {ts}", "", "| warunek | RGB-only | pełny model | pełny − RGB-only (wkład echa+materiału+uwagi, sparowany po ziarnie) |", "|---|---|---|---|"]
    for c in "ADB": L.append(f"| {c} | {ms(R[c])} | {ms(F[c])} | {pt(F[c], R[c])} |")
    L += ["", "| kontrast | RGB-only | pełny model |", "|---|---|---|"]
    for lab, (x, y) in {"D − A": ("D", "A"), "B − D": ("B", "D"), "B − A": ("B", "A")}.items():
        L.append(f"| {lab} | {pt(R[x], R[y])} | {pt(F[x], F[y])} |")
    L.append("")
L += ["Odczyt: jeśli `D − A` występuje także w RGB-only, to część „efektu gęstości kątowej” w modelu pełnym pochodzi z różnorodności obrazów (orientacji kamery), "
      "a nie z echa; różnica kontrastów (pełny − RGB-only) szacuje część przypisywalną echu. To są hipotezy do rozstrzygnięcia na liczbach powyżej (3 ziarna, df = 2)."]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8"); print("zapisano", OUT)
