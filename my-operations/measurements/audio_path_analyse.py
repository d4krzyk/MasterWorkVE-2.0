#!/usr/bin/env python3
"""KROK 2/2: czy 1 symulacja audio na render jest rownowazna 2 symulacjom?

Porownuje sigma_1 miedzy sciezkami DWOMA estymatorami:
  - polowkowym  (ten, ktorego uzywa generator — dla zgodnosci z regula na N)
  - wariancyjnym (sigma^2 = srednia po komorkach z Var po renderach; n-1 stopni
    swobody na komorke zamiast 1, wiec DUZO dokladniejszy)

Sprawdza takze rozgrzewke w blokach po 10 renderow — to ona wyjasnia, dlaczego
pierwsze podejscie estymatorem polowkowym dalo pozornie niepokojace +7 i +18 %.

Wynik (2026-07-29): estymator wariancyjny daje roznice +1.3 % (0.51 SE) i
+3.0 % (0.75 SE) — obie ponizej progu istotnosci. Wplyw na N: 30->31 i 4->4.

WYMAGA: outputs/measurements/paths_specs.npz (z audio_path_render.py)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.2, §2.6 | Dokument: GENERATOR_PARAMS.md §4.3
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
MEAS_OUT = Path(__file__).resolve().parents[2] / "outputs/measurements"

from echo_core.params import N_MIN, SIGNAL_10DEG, TARGET_SNR
import sys
from pathlib import Path

sys.path.insert(0, "/home/d4krzyk/Dokumenty/MasterWorkVE/my-operations")

D = np.load(str(MEAS_OUT / "paths_specs.npz"))
POSITIONS = [("office_1", 33), ("frl_apartment_5", 186)]
PATHS = ("podwojna", "pojedyncza")
ANGLES = (0, 10)


def sigma_direct(specs):
    """sqrt(srednia po komorkach z wariancji po renderach) = szum 1 renderu."""
    return float(np.sqrt(np.var(specs.astype(np.float64), axis=0, ddof=1).mean()))


def sigma_halfsplit(specs):
    h = len(specs) // 2
    a = specs[:h].mean(axis=0, dtype=np.float64)
    b = specs[h:2 * h].mean(axis=0, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(2.0) * np.sqrt(h))


# --- 1. Czy widac rozgrzewke? ------------------------------------------------
print(f"{'='*100}\n  KONTROLA ROZGRZEWKI: szum w blokach po 10 renderow (kat 0 st.)\n{'='*100}")
print(f"  {'pozycja':<22}{'sciezka':<12}{'r0-9':>10}{'r10-19':>10}{'r20-29':>10}{'r30-39':>10}")
for scene, loc in POSITIONS:
    for path in PATHS:
        s = D[f"{scene}|{loc}|{path}|0"]
        blocks = [sigma_direct(s[i:i + 10]) for i in range(0, 40, 10)]
        print(f"  {f'{scene}/{loc}':<22}{path:<12}" + "".join(f"{b:>10.5f}" for b in blocks))

# --- 2. Porownanie sciezek dokladnym estymatorem ------------------------------
print(f"\n{'='*100}\n  SIGMA_1 — estymator wariancyjny (M-1 st. swobody) vs podzial na polowki\n{'='*100}")
print(f"  {'pozycja':<22}{'sciezka':<12}{'kat':>5}{'wariancyjny':>14}{'polowkowy':>12}")
sig = {}
for scene, loc in POSITIONS:
    for path in PATHS:
        vals = []
        for a in ANGLES:
            s = D[f"{scene}|{loc}|{path}|{a}"]
            sd, sh = sigma_direct(s), sigma_halfsplit(s)
            vals.append(sd)
            print(f"  {f'{scene}/{loc}':<22}{path:<12}{a:>5}{sd:>14.5f}{sh:>12.5f}")
        sig[(scene, path)] = np.array(vals)

# Niepewnosc estymatora wariancyjnego z replikatow: 2 rozlaczne polowki x 2 katy
print(f"\n{'='*100}\n  ROZNICA MIEDZY SCIEZKAMI (estymator wariancyjny)\n{'='*100}")
for scene, loc in POSITIONS:
    reps = {}
    for path in PATHS:
        r = []
        for a in ANGLES:
            s = D[f"{scene}|{loc}|{path}|{a}"]
            r += [sigma_direct(s[:20]), sigma_direct(s[20:])]
        reps[path] = np.array(r)
    a_, b_ = reps["podwojna"], reps["pojedyncza"]
    pooled = np.concatenate([a_ - a_.mean(), b_ - b_.mean()])
    sd_rep = float(np.sqrt((pooled ** 2).sum() / 6))     # 2x4 replikaty - 2 srednie
    se_point = sd_rep / 2.0                              # punkt z 4x wiecej renderow
    se_diff = se_point * np.sqrt(2)
    pa, pb = sig[(scene, "podwojna")].mean(), sig[(scene, "pojedyncza")].mean()
    d = pb - pa
    print(f"\n  --- {scene}/{loc} ---")
    print(f"  replikaty podwojna   : {' '.join(f'{v:.5f}' for v in a_)}")
    print(f"  replikaty pojedyncza : {' '.join(f'{v:.5f}' for v in b_)}")
    print(f"  sigma_1  {pa:.5f} -> {pb:.5f}   roznica {d:+.5f} ({d/pa*100:+.1f} %)")
    print(f"           SD replikatu {sd_rep:.5f}, SE roznicy {se_diff:.5f}  ->  "
          f"{abs(d)/se_diff:.2f} SE")
    na = int(np.ceil((TARGET_SNR * pa / SIGNAL_10DEG) ** 2))
    nb = int(np.ceil((TARGET_SNR * pb / SIGNAL_10DEG) ** 2))
    print(f"           N z reguly: {na} -> {nb}  (koszt {nb/na:.2f}x renderow, "
          f"przy 2.0x szybszym renderze -> netto {2.0*na/nb:.2f}x)")


# --- ENERGIA I SYGNAL 10 st. — kryterium glowne rownowaznosci -----------------
# Energia spektrogramu jest NAJMNIEJ zaszumiona z mierzonych wielkosci (srednia po
# 80 renderach x 85 324 komorek), wiec to ona rozstrzyga o rownowaznosci sciezek.
# sigma_1 jest istotne dla kosztu (przez N), ale ma wiekszy rozrzut.
print(f"\n{'='*104}\n  ENERGIA I SYGNAL 10 st.\n{'='*104}")
try:
    from scipy.stats import mannwhitneyu
except ImportError:                      # pragma: no cover
    mannwhitneyu = None


def _mean(x):
    return np.mean(x, axis=0, dtype=np.float32)


def _rmse2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


for scene, loc in POSITIONS:
    out = {}
    for path in PATHS:
        spec = {a: D[f"{scene}|{loc}|{path}|{a}"] for a in ANGLES}
        energies = np.concatenate([spec[a].mean(axis=(1, 2, 3)) for a in ANGLES])
        m = len(spec[0])
        sm = 0.5 * (sigma_direct(spec[0]) + sigma_direct(spec[10])) / np.sqrt(m)
        raw = _rmse2(_mean(spec[0]), _mean(spec[10]))
        # RMSE(est_0, est_10)^2 = sygnal^2 + 2*sigma_M^2 — dekompozycja obowiazkowa,
        # porownanie surowych RMSE dwoch zaszumionych estymat zawyzaloby sygnal.
        signal = float(np.sqrt(max(raw ** 2 - 2 * sm ** 2, 0.0)))
        out[path] = dict(energy=float(energies.mean()),
                         se=float(energies.std(ddof=1) / np.sqrt(energies.size)),
                         energies=energies, signal=signal)

    a, b = out["podwojna"], out["pojedyncza"]
    de = b["energy"] - a["energy"]
    sed = float(np.hypot(a["se"], b["se"]))
    print(f"\n  --- {scene}/{loc} ---")
    print(f"  energia      {a['energy']:.6f} -> {b['energy']:.6f}   roznica {de:+.3e} "
          f"({de/a['energy']*100:+.3f} %)")
    print(f"               SE roznicy {sed:.3e}  ->  {abs(de)/sed:.2f} SE")
    if mannwhitneyu is not None:
        _u, p = mannwhitneyu(b["energies"], a["energies"], alternative="two-sided")
        print(f"               Mann-Whitney p = {p:.3f} (n={len(a['energies'])} na sciezke)")
    ds = b["signal"] - a["signal"]
    print(f"  sygnal 10st  {a['signal']:.5f} -> {b['signal']:.5f}   "
          f"({ds/a['signal']*100:+.1f} %)   [SIGNAL_10DEG={SIGNAL_10DEG}, zakres 0.0639-0.0662]")

print(f"\n  Prog przyjety: |roznica| > 2 SE = rozbieznosc systematyczna.")
