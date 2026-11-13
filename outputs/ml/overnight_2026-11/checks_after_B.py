#!/usr/bin/env python
"""checks_after_B.md: (a) ostatnia cwiartka vs warunek, (b) podzial probe_aux, (c) liczby kroku 7. Tylko odczyt; fail-if-exists."""
import json, re, sys, collections
from pathlib import Path
import numpy as np
from scipy import stats
REPO = Path(__file__).resolve().parents[3]; H = Path(__file__).resolve().parent; OUT = H / "checks_after_B.md"
if OUT.exists(): sys.exit("checks_after_B.md juz istnieje")
L = ["# Kontrole po części B (tylko odczyt)", ""]
# (a)
md = (H / "plateau_recomputed.md").read_text(encoding="utf-8")
by = collections.OrderedDict(); late = []
for l in md.splitlines():
    m = re.match(r"\| ([A-Z]+[0-9]*)_seed(\d) \|", l)
    if not m: continue
    c = [x.strip() for x in l.strip("|").split("|")]
    cond, sd, bs = m.group(1), int(m.group(2)), int(c[8])
    by.setdefault(cond, []).append((sd, bs))
    if bs >= 31000: late.append((cond, sd, bs))
L += ["## a) Przebiegi z najlepszym krokiem w ostatniej ćwiartce (≥ 31 000)", "",
      f"Razem {len(late)} z {sum(len(v) for v in by.values())} (źródło: `plateau_recomputed.md`, kolumna `najl_krok`).", "",
      "| warunek | przebiegi z krokiem ≥ 31 000 / wszystkie | ziarno: najlepszy krok |", "|---|---|---|"]
for cond, v in by.items():
    n = sum(1 for _, b in v if b >= 31000)
    L.append(f"| {cond} | {n}/{len(v)} | " + ", ".join(f"s{s}: {b}" for s, b in sorted(v)) + " |")
L += ["", "Lista: " + ", ".join(f"{c}_seed{s} ({b})" for c, s, b in late) + ".", ""]
# (b)
st = json.loads((REPO / "outputs/ml/probing/probe_aux_random_seed0/status.json").read_text())
ntr, nev = st["n_train"], st["n_eval"]
L += ["## b) Podział train/eval w `probe_aux`", "",
      "- Plik: `my-operations/ml/pretext_model/probe.py`, funkcja `_aux_split` (l. 383–402), wywołana w `cmd_aux` (l. 445: `is_eval = _aux_split(scene_idx, loc_id)`).",
      "- Mechanizm: klucz `scene_idx*100000 + loc_id` (l. 397) → `np.unique` → z unikalnych LOKALIZACJI losowane `round(0,2·N)` bez zwracania (`default_rng(20260817)`, l. 399–401) → maska `k in eval_locs` dla każdej próbki (l. 402). Podział jest **po lokalizacjach**, nie po próbkach (docstring l. 384 to potwierdza).",
      f"- Zgodność z `status.json`: n_train = {ntr} = {ntr//36}·36, n_eval = {nev} = {nev//36}·36; {ntr//36} + {nev//36} = {ntr//36 + nev//36} lokalizacji = 1374 treningowych (`angle_diversity.json`); round(0,2·1374) = {round(0.2*1374)} = {nev//36} ✓.",
      "- Lokalizacje występujące jednocześnie w obu zbiorach: **0** (maska zależy wyłącznie od klucza lokalizacji, więc wszystkie 36 próbek lokalizacji trafia do tej samej strony). Sprawdzone z kodu i arytmetyki; nie odtwarzałem cech na GPU.",
      "- Zastrzeżenie: eval to inne lokalizacje, ale z tych samych scen i pomieszczeń co trening sondy, i to zbiór treningowy głębi (nie val/test) — to zamierzone (docstring l. 385–395: klasą jest tożsamość sceny).", ""]
# (c)
def ev(p, ts):
    return json.loads(Path(p).read_text())["test_sets"][ts]["overall"]["all"]["RMSE"]
S = (0, 1, 2)
def col(kind, c, ts):
    return np.array([ev(H / f"rgbonly/RGBONLY_{c}_seed{s}/eval.json" if kind == "rgb" else REPO / f"outputs/ml/eval/{c}_seed{s}/eval.json", ts) for s in S])
def ms(v): return f"{v.mean():.5f} ± {v.std(ddof=1):.5f}"
def pt(a, b):
    d = a - b; t = stats.ttest_1samp(d, 0.0)
    return f"{d.mean():+.5f} ± {d.std(ddof=1):.5f} (p = {t.pvalue:.4f})"
L += ["## c) Krok 7 — dokładne liczby (RMSE [m]; średnia ± sd po 3 ziarnach, sd z ddof = 1; p = `ttest_1samp` na różnicach po ziarnach, df = 2)", ""]
share = {}
for ts in ("test@36", "test@4"):
    R = {c: col("rgb", c, ts) for c in "ADB"}; F = {c: col("full", c, ts) for c in "ADB"}
    L += [f"### {ts}", "", "| warunek | RGB-only | model pełny | wartości RGB-only s0/s1/s2 | wartości pełne s0/s1/s2 |", "|---|---|---|---|---|"]
    for c in "ADB":
        L.append(f"| {c} | {ms(R[c])} | {ms(F[c])} | {' / '.join(f'{x:.5f}' for x in R[c])} | {' / '.join(f'{x:.5f}' for x in F[c])} |")
    L += ["", "Kontrasty (X − Y, ujemne = X lepsze):", "", "| kontrast | RGB-only | model pełny |", "|---|---|---|"]
    for lab, (x, y) in {"D − A": ("D", "A"), "B − D": ("B", "D"), "B − A": ("B", "A")}.items():
        L.append(f"| {lab} | {pt(R[x], R[y])} | {pt(F[x], F[y])} |")
    L += ["", "Udział (D − A)/(B − A) per ziarno (para (D_s − A_s)/(B_s − A_s); ziarna sparowane indeksem):", "", "| ziarno | RGB-only | model pełny |", "|---|---|---|"]
    sr = (R["D"] - R["A"]) / (R["B"] - R["A"]); sf = (F["D"] - F["A"]) / (F["B"] - F["A"])
    for i, s in enumerate(S): L.append(f"| {s} | {sr[i]:.3f} | {sf[i]:.3f} |")
    L.append(f"| średnia ± sd | {ms(sr)} | {ms(sf)} |")
    mr = (R["D"].mean() - R["A"].mean()) / (R["B"].mean() - R["A"].mean()); mf = (F["D"].mean() - F["A"].mean()) / (F["B"].mean() - F["A"].mean())
    L += [f"| iloraz średnich | {mr:.3f} | {mf:.3f} |", ""]
Rb = col("rgb", "B", "test@36"); se = ev(REPO / "outputs/ml/eval/SE_seed0/eval.json", "test@36")
Fb = ev(REPO / "outputs/ml/eval/B_seed0/eval.json", "test@36")
L += ["### RGB-only B vs SE_seed0 (test@36)", "",
      f"- RGB-only B: ziarna {' / '.join(f'{x:.5f}' for x in Rb)}, średnia {Rb.mean():.5f}.",
      f"- SE_seed0 (model pełny z potasowanym echem, ziarno 0): {se:.5f}; B_seed0 (model pełny): {Fb:.5f}.",
      f"- SE_seed0 − RGB-only B_seed0 = {se - Rb[0]:+.5f}; SE_seed0 − średnia RGB-only B = {se - Rb.mean():+.5f} (sd RGB-only B = {Rb.std(ddof=1):.5f}, czyli {(se - Rb.mean())/Rb.std(ddof=1):+.1f} sd). Przy n = 1 dla SE nie ma testu istotności.",
      "- Uwaga: SE_seed0 uczono na wszystkich 36 orientacjach (jak B), więc to właściwe porównanie 'echo bez informacji kątowej' vs 'brak echa'.", ""]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8"); print("zapisano", OUT)
