#!/usr/bin/env python3
"""ANALIZA CENSUS: pelny rozklad N_raw dla wszystkich 1740 lokalizacji.

Odpowiada na kluczowe pytanie projektu: czy N_MAX pokrywa CALY zbior, czy tylko
zmierzona probke? Wczesniejsze uzasadnienia opieraly sie na 12, potem 52 pozycjach
dobieranych po ustalonych ulamkach listy — census mierzy wszystkie 1740.

Wynik (2026-07-29):
    sigma_1  mediana 0.05830, zakres 0.02530-0.13451
    N_raw    mediana 11, srednia 11.80, zakres 2-54
    N_raw > 40: 7 lokalizacji (0.402 %)   <- ekstrapolacja z 52 pozycji mowila ZERO
    N_raw > 64: 0 lokalizacji
Powod chybienia ekstrapolacji: gorace miejsce w apartment_0 lezy przy loc_id
285-310, czyli okolo ulamka 0.9 listy — poza probkowanymi 0.20 i 0.75.

WYMAGA: outputs/probe_census/*.csv (z `generate_echo_dataset.py --probe-only`)
PRODUKUJE: outputs/diagnose_rlr_noise_out/probe_census.png (wersjonowany)
           outputs/probe_census/_all_scenes.csv (scalony)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.3, §2.4 | Dokument: GENERATOR_PARAMS.md §3.2
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_core.params import (N_MAX, N_MIN, N_PROBE, SIGNAL_10DEG, TARGET_SNR,
                              WARMUP_DISCARD)
from echo_core.paths import probe_census_csv, scene_h5
from echo_core.scenes import SCENE_ORDER

frames = []
for s in SCENE_ORDER:
    p = probe_census_csv(s)
    if not p.exists():
        print(f"  BRAK: {p}")
        continue
    frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=["scene", "loc_id"], keep="last")

SIGMA_NMAX = float(np.sqrt(N_MAX) * SIGNAL_10DEG / TARGET_SNR)
print(f"{'='*100}\n  CENSUS SONDY — {len(df)} lokalizacji\n{'='*100}")
exp = {s: 0 for s in SCENE_ORDER}
import pickle
obs = pickle.load(open(Path(__file__).resolve().parents[2] / "scenes_ve_metadata_locations/scene_observations_128.pkl", "rb"))
for s in SCENE_ORDER:
    exp[s] = len({k[0] for k in obs[s].keys()})
del obs
missing = {s: exp[s] - int((df.scene == s).sum()) for s in SCENE_ORDER}
bad = {s: v for s, v in missing.items() if v != 0}
print(f"  kompletnosc: {len(df)} / {sum(exp.values())}"
      + (f"   BRAKI: {bad}" if bad else "   (komplet)"))

n = df.n_raw.values
s1 = df.sigma_1.values
print(f"\n  sigma_1 : mediana {np.median(s1):.5f}  srednia {s1.mean():.5f}  "
      f"zakres {s1.min():.5f}-{s1.max():.5f}")
print(f"  N_raw   : mediana {int(np.median(n))}  srednia {n.mean():.2f}  "
      f"zakres {n.min()}-{n.max()}")

print(f"\n{'='*100}\n  PELNY ROZKLAD N_raw (1740 lokalizacji)\n{'='*100}")
bins = [(1, 5), (6, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 30),
        (31, 40), (41, 48), (49, 64), (65, 10**6)]
for lo, hi in bins:
    m = (n >= lo) & (n <= hi)
    c = int(m.sum())
    lbl = f"{lo}-{hi}" if hi < 10**5 else f">={lo}"
    bar = "#" * int(round(60 * c / len(n)))
    flag = ""
    if lo >= 41:
        flag = "   <-- POWYZEJ N_MAX=40"
    if lo >= 65:
        flag = "   <-- POWYZEJ 64"
    print(f"  N_raw {lbl:>8} | {bar:<60} {c:>5} ({100*c/len(n):5.2f} %){flag}")

over40 = df[df.n_raw > 40].sort_values("n_raw", ascending=False)
over64 = df[df.n_raw > 64].sort_values("n_raw", ascending=False)
print(f"\n{'='*100}\n  PRZEKROCZENIA PROGOW\n{'='*100}")
print(f"  N_raw > 40 (N_MAX): {len(over40)} / {len(df)}  ({100*len(over40)/len(df):.3f} %)")
print(f"  N_raw > 64        : {len(over64)} / {len(df)}  ({100*len(over64)/len(df):.3f} %)")
print(f"  N_raw > 24        : {int((n > 24).sum())} / {len(df)}  "
      f"({100*(n > 24).mean():.2f} %)")
if len(over40):
    print(f"\n  LISTA N_raw > 40:")
    print(f"  {'scena':<20}{'loc_id':>8}{'sigma_1':>11}{'N_raw':>8}")
    for _, r in over40.iterrows():
        print(f"  {r.scene:<20}{int(r.loc_id):>8}{r.sigma_1:>11.5f}{int(r.n_raw):>8}")

top = df.nlargest(10, "sigma_1")
print(f"\n{'='*100}\n  10 NAJGLOSNIEJSZYCH LOKALIZACJI W CALYM ZBIORZE\n{'='*100}")
print(f"  {'scena':<20}{'loc_id':>8}{'sigma_1':>11}{'N_raw':>8}{'zapas do progu':>16}")
for _, r in top.iterrows():
    print(f"  {r.scene:<20}{int(r.loc_id):>8}{r.sigma_1:>11.5f}{int(r.n_raw):>8}"
          f"{SIGMA_NMAX/r.sigma_1:>15.2f}x")

print(f"\n{'='*100}\n  PER SCENA\n{'='*100}")
print(f"  {'scena':<20}{'lok.':>6}{'sigma_1 med':>13}{'sigma_1 max':>13}"
      f"{'N med':>7}{'N max':>7}{'>40':>6}{'>24':>6}")
g = df.groupby("scene")
rows = []
for s in SCENE_ORDER:
    d = g.get_group(s)
    rows.append((s, len(d), d.sigma_1.median(), d.sigma_1.max(),
                 int(d.n_raw.median()), int(d.n_raw.max()),
                 int((d.n_raw > 40).sum()), int((d.n_raw > 24).sum())))
for r in sorted(rows, key=lambda x: -x[2]):
    print(f"  {r[0]:<20}{r[1]:>6}{r[2]:>13.5f}{r[3]:>13.5f}{r[4]:>7}{r[5]:>7}{r[6]:>6}{r[7]:>6}")

print(f"\n{'='*100}\n  POROWNANIE Z WCZESNIEJSZA EKSTRAPOLACJA\n{'='*100}")
print(f"  ekstrapolacja z 52 pozycji przewidywala 0.066 % PROBEK przy limicie (~41 z 62 640)")
print(f"  census (pelny, 1740 lokalizacji):")
print(f"    lokalizacji z N_raw > 40      : {len(over40)} ({100*len(over40)/len(df):.3f} %)")
print(f"    probek w tych lokalizacjach   : {36*len(over40)} z 62 640 "
      f"({100*36*len(over40)/62640:.3f} %)")
print(f"  UWAGA: to sa rozne wielkosci. Ekstrapolacja szacowala probki, ktorych sigma")
print(f"  PRZY SPRAWDZENIU przekroczy prog; census mierzy lokalizacje, ktorych SONDA")
print(f"  zada wiecej niz N_MAX. Pierwsza dotyczy pojedynczych orientacji, druga calych")
print(f"  lokalizacji (36 probek kazda).")

# rozklad + budzet
print(f"\n{'='*100}\n  BUDZET Z PELNEGO ROZKLADU\n{'='*100}")
import h5py
with h5py.File(scene_h5("office_1"), "r", locking=False) as f:
    w = f["written"][:].astype(bool)
    SPR = float(f.attrs["seconds_per_render"])
    OV = float(f["n_total"][:][w].sum() / f["n_planned"][:][w].sum())
npl = np.clip(n, N_MIN, N_MAX)
renders = float((N_PROBE + 36 * npl - np.minimum(N_PROBE, npl)).sum()) * OV + 18 * WARMUP_DISCARD
print(f"  srednie N po clamp [{N_MIN},{N_MAX}] : {npl.mean():.2f}")
print(f"  renderow lacznie                : {renders:,.0f}".replace(",", " "))
print(f"  czas przy {SPR:.4f} s/render      : {renders*SPR/3600:.1f} h")
print(f"  (narzut petli weryfikacyjnej {OV:.4f}x uwzgledniony, rozgrzewka 18x{WARMUP_DISCARD})")

# --- wykres ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ax = axes[0]
ax.hist(n, bins=np.arange(0.5, max(n.max(), 45) + 1.5, 1), color="tab:blue", edgecolor="none")
ax.axvline(N_MAX, color="tab:red", ls="--", lw=1.5, label=f"N_MAX = {N_MAX}")
ax.axvline(24, color="tab:orange", ls=":", lw=1.2, label="porzucony prog 24")
ax.set_xlabel("N_raw (przed clamp)")
ax.set_ylabel("lokalizacji")
ax.set_title(f"Rozkład N_raw — wszystkie {len(df)} lokalizacji", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
order = [r[0] for r in sorted(rows, key=lambda x: -x[2])]
data = [g.get_group(s).sigma_1.values for s in order]
ax.boxplot(data, vert=False, labels=order, widths=0.6, showfliers=True,
           flierprops=dict(marker=".", markersize=4))
ax.axvline(SIGMA_NMAX, color="tab:red", ls="--", lw=1.5,
           label=f"próg przy N_MAX: {SIGMA_NMAX:.4f}")
ax.set_xlabel("sigma_1")
ax.set_title("Podłoga szumu per scena (pełny census)", fontsize=10)
ax.legend(fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "outputs/diagnose_rlr_noise_out/probe_census.png"
fig.savefig(out, dpi=130)
print(f"\n  Wykres -> {out}")

df.to_csv(Path(__file__).resolve().parents[2] / "outputs/probe_census/_all_scenes.csv", index=False)
print(f"  Scalony CSV -> outputs/probe_census/_all_scenes.csv")
