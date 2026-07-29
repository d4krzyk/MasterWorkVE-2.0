#!/usr/bin/env python3
"""DOWOD: estymator polowkowy ma SD 4-6 %, a jego dokladnosc NIE poprawia sie z n.

Pytanie: GENERATOR_PARAMS.md §3.4 uzasadnial weryfikacje po fakcie zdaniem
"oszacowanie sigma_1 z 8 renderow ma ~10 % bledu" — liczba oszacowana, nigdy
niezmierzona. Ile wynosi naprawde?

Metoda: bootstrap na renderach ze STANU USTALONEGO (po odrzuceniu rozgrzewki),
referencja z estymatora wariancyjnego po wszystkich 80 renderach.

Wynik (2026-07-29): SD 5.5 / 4.2 / 4.5 % przy n=8 — czyli sonda jest DWUKROTNIE
dokladniejsza, niz zakladano, i praktycznie nieobciazona (-0.1 %).

Wynik dodatkowy, wazniejszy: SD jest IDENTYCZNE przy n = 8, 20 i 40. Powod jest
strukturalny — RMSE^2*h/2 jest estymatorem sigma^2 o JEDNYM stopniu swobody na
komorke, niezaleznie od h. Ogranicza go efektywna liczba niezaleznych komorek
spektrogramu (~600-1200 z 85 324), nie liczba renderow. Estymator wariancyjny ma
n-1 stopni swobody na komorke i poprawia sie z n.

WYMAGA: outputs/measurements/warmup_specs.npz (z simulator_warmup.py)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.6 | Dokument: GENERATOR_PARAMS.md §3.4
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/d4krzyk/Dokumenty/MasterWorkVE/my-operations")
from echo_core import audio, spectrogram
from echo_core.params import (INDIRECT_RAY_COUNT, N_MAX, N_MIN, N_PROBE, SENSOR_HEIGHT,
                              SIGNAL_10DEG, TARGET_SNR, THREAD_COUNT, WARMUP_DISCARD)
from echo_core.paths import CHIRP_PATH, MATERIAL_CONFIG, OUT_ROOT, probe_census_csv, scene_mesh
from echo_core.renderer import Renderer
from echo_core.runtime import setup_logging
from echo_core.scenes import load_scene_locations
from echo_core.store import SPEC_SHAPE

MEAS_OUT = Path(__file__).resolve().parents[2] / "outputs/measurements"
MEAS_OUT.mkdir(parents=True, exist_ok=True)

D = np.load(str(MEAS_OUT / "warmup_specs.npz"))
WARMUP = 20          # rendery odrzucone jako rozgrzewka (Blok A)
N_BOOT = 4000
RNG = np.random.default_rng(20260729)

CASES = [("office_1", "A", 33), ("frl_apartment_5", "A", 186), ("room_0", "A", 43)]


def sigma_var(specs):
    """Estymator wariancyjny: sqrt(srednia po komorkach z Var_po_renderach)."""
    return float(np.sqrt(np.var(specs.astype(np.float64), axis=0, ddof=1).mean()))


def sigma_halfsplit(specs):
    """Dokladnie ten estymator, ktorego uzywa generator (§3.2)."""
    h = len(specs) // 2
    a = specs[:h].mean(axis=0, dtype=np.float64)
    b = specs[h:2 * h].mean(axis=0, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(2.0) * np.sqrt(h))


print(f"{'='*100}")
print(f"  BLAD SONDY (n=8, estymator polowkowy 4+4) — bootstrap {N_BOOT}x na renderach")
print(f"  ze stanu ustalonego (odrzucone pierwsze {WARMUP} renderow)")
print(f"{'='*100}")

summary = []
for scene, side, loc in CASES:
    pool = D[f"{scene}|{side}"][WARMUP:]          # 80 renderow stanu ustalonego
    ref = sigma_var(pool)
    # niepewnosc referencji: dwie rozlaczne polowki po 40
    ref_halves = [sigma_var(pool[:40]), sigma_var(pool[40:])]
    ref_unc = abs(ref_halves[0] - ref_halves[1]) / 2 / np.sqrt(2)   # SE z 2 replikatow

    est8 = np.array([sigma_halfsplit(pool[RNG.choice(len(pool), 8, replace=False)])
                     for _ in range(N_BOOT)])
    rel = est8 / ref
    sd_rel = float(rel.std(ddof=1))
    bias = float(rel.mean() - 1)
    p5, p50, p95 = (float(np.percentile(rel, q)) for q in (5, 50, 95))

    # propagacja na N: N ~ sigma^2
    n_rel = rel ** 2
    sd_n = float(n_rel.std(ddof=1))

    print(f"\n  --- {scene}/{loc} ---")
    print(f"  sigma_1 referencyjna (80 renderow, estymator wariancyjny): {ref:.5f} "
          f"+- {ref_unc:.5f} ({100*ref_unc/ref:.1f} %)")
    print(f"  sonda n=8, estymator polowkowy:")
    print(f"    SD wzgledem referencji : {100*sd_rel:.1f} %")
    print(f"    obciazenie (srednia-1) : {100*bias:+.1f} %")
    print(f"    percentyle 5/50/95     : {p5:.3f} / {p50:.3f} / {p95:.3f} "
          f"(x referencja)")
    print(f"  propagacja na N (N ~ sigma^2):")
    print(f"    SD N                   : {100*sd_n:.1f} %")
    print(f"    percentyle 5/50/95     : {p5**2:.3f} / {p50**2:.3f} / {p95**2:.3f}")
    summary.append(dict(scene=scene, ref=ref, sd_rel=sd_rel, bias=bias, sd_n=sd_n,
                        p5=p5, p95=p95, pool=pool))

print(f"\n{'='*100}")
print("  SYMULACJA CALEJ REGULY: jaki odsetek probek trafi ponizej progu SNR?")
print(f"{'='*100}")
print("""  Odtwarzamy dokladnie logike generatora na prawdziwych renderach:
    1. losuj 8 renderow -> sigma_1 (polowkowy 4+4) -> N = clamp(ceil((3.5*s/0.0644)^2), 6, 40)
    2. losuj N renderow -> snr_probe = 0.0644*sqrt(N)/sigma_1(polowkowy N/2+N/2)
    3. dorenderowanie potrzebne <=> snr_probe < 3.5""")

for s in summary:
    pool = s["pool"]
    need, ns = 0, []
    trials = 0
    for _ in range(N_BOOT):
        idx = RNG.permutation(len(pool))
        probe = pool[idx[:8]]
        sp = sigma_halfsplit(probe)
        n_raw = int(np.ceil((TARGET_SNR * sp / SIGNAL_10DEG) ** 2)) if sp > 0 else N_MIN
        n = int(min(max(n_raw, N_MIN), N_MAX))
        if 8 + n > len(pool):
            continue
        trials += 1
        final = pool[idx[8:8 + n]]
        sf = sigma_halfsplit(final)
        snr = SIGNAL_10DEG * np.sqrt(n) / sf if sf > 0 else np.inf
        need += snr < TARGET_SNR
        ns.append(n)
    ns = np.array(ns)
    print(f"\n  {s['scene']:<18} N: mediana {int(np.median(ns))}, srednia {ns.mean():.1f}, "
          f"zakres {ns.min()}-{ns.max()}   ({trials} prob)")
    print(f"  {'':<18} PRZEWIDYWANY odsetek dorenderowan: {100*need/trials:.1f} %")

print(f"\n{'='*100}")
print(f"  ZAOBSERWOWANE na office_1 (pelna scena, 576 probek): 41.0 %")
print(f"{'='*100}")
