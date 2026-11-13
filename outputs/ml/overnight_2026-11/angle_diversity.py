#!/usr/bin/env python
"""Roznorodnosc orientacji w zbiorze treningowym (A1): ile roznych orientacji wystepuje w zbiorze i jaki
ulamek lokalizacji zawiera kazda z nich, dla A, D i EK6/9/12/18. Tylko CPU, dane z kodu (angles.py, splity).
Wynik: angle_diversity.json/.md w tym katalogu (fail-if-exists)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "my-operations"))
from ml.dataset import angles                      # noqa: E402
from ml.dataset.splits import load_splits           # noqa: E402
from ml.matrix.experiments import CONDITIONS_BY_ID  # noqa: E402

OUT = Path(__file__).resolve().parent
for n in ("angle_diversity.json", "angle_diversity.md"):
    if (OUT / n).exists():
        sys.exit(f"{n} juz istnieje")
sp = load_splits()
locs = [(sc, int(l)) for sc, ls in sorted(sp.train.items()) for l in ls]
GRID = {"A": 4, "D": 36, "EK6": 6, "EK9": 9, "EK12": 12, "EK18": 18}
res = {}
for c, K in GRID.items():
    cd = CONDITIONS_BY_ID[c]
    cnt = {}
    n_per = []
    for sc, l in locs:
        a = angles.select_angles(cd.angle_subset, scene=sc, location_id=l, seed=cd.angle_seed)
        n_per.append(len(a))
        for x in a:
            cnt[int(x)] = cnt.get(int(x), 0) + 1
    fr = np.array([cnt[k] / len(locs) for k in sorted(cnt)])
    res[c] = {"angle_subset": cd.angle_subset, "angle_seed": cd.angle_seed, "K_siatki": K,
              "lokalizacji": len(locs), "probek": int(sum(n_per)), "katow_na_lokalizacje": sorted(set(n_per)),
              "rozne_orientacje_w_zbiorze": len(cnt), "ulamek_lokalizacji_z_orientacja": {
                  "srednia": float(fr.mean()), "min": float(fr.min()), "max": float(fr.max()),
                  "oczekiwane_4_przez_K": 4 / K}}
(OUT / "angle_diversity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
L = ["# Różnorodność orientacji w zbiorze treningowym (1374 lokalizacji)", "",
     "Źródło: `angles.select_angles` (hash `f\"{angle_seed}|{scene}|{location_id}\"`) i podział treningowy; `angle_diversity.py`.", "",
     "| warunek | podzbiór | K siatki | próbek | różnych orientacji w zbiorze | ułamek lokalizacji z daną orientacją: średnia / min / maks. | oczekiwane 4/K |",
     "|---|---|---|---|---|---|---|"]
for c, r in res.items():
    u = r["ulamek_lokalizacji_z_orientacja"]
    L.append(f"| {c} | `{r['angle_subset']}` (seed {r['angle_seed']}) | {r['K_siatki']} | {r['probek']} | **{r['rozne_orientacje_w_zbiorze']}** | "
             f"{u['srednia']:.4f} / {u['min']:.4f} / {u['maks'] if 'maks' in u else u['max']:.4f} | {u['oczekiwane_4_przez_K']:.4f} |")
(OUT / "angle_diversity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[-8:]))
