#!/usr/bin/env python3
"""DOWOD: rozgrzewka Simulatora jest wlasnoscia KONSTRUKCJI, nie pozycji agenta.

Pytanie 1: czy pierwsze rendery w swiezej instancji roznia sie tylko szumem, czy
takze srednia (energia)?
Pytanie 2 (kluczowe): czy efekt dotyczy kazdej lokalizacji, czy tylko pierwszej
w scenie? Od tego zalezy, czy skala problemu to 18, czy 1740 lokalizacji.

Metoda: swieza instancja -> 100 renderow na pozycji A -> przeniesienie agenta na
pozycje B W TEJ SAMEJ INSTANCJI -> 30 renderow. Jesli po przeniesieniu pierwszy
blok znowu jest glosniejszy, efekt jest per pozycja.

NIE usredniamy po powtorzeniach swiezych instancji: E1 dowiodl, ze swiezy
Simulator odtwarza IDENTYCZNA sekwencje RNG, wiec R konstrukcji daloby R
identycznych wynikow. Niezaleznosc bierze sie z roznych pozycji i scen.

Szum liczony ESTYMATOREM WARIANCYJNYM (sigma^2 = srednia po komorkach z Var po
renderach), nie polowkowym — ten ma sufit dokladnosci ~5 % niezalezny od liczby
renderow (patrz probe_estimator_accuracy.py).

Wynik (2026-07-29, 3 sceny):
    blok 1 (r0-9) wyzszy o +11.4 / +19.9 / +10.4 % wzgledem stanu ustalonego
    po przeniesieniu agenta: stosunek pierwszy/pozostale = 1.004 / 0.999 / 0.993
    -> efekt PER KONSTRUKCJA, dotyczy 18 lokalizacji, nie 1740
    energia: -0.88 / -0.77 / -0.68 % = 1.75 / 1.36 / 1.14 SE (ponizej istotnosci)

PRODUKUJE: outputs/diagnose_rlr_noise_out/warmup_simulator.png (wersjonowany)
           outputs/measurements/warmup_specs.npz (~115 MiB, gitignored)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.7 | Dokument: GENERATOR_PARAMS.md §2
"""
import sys, time
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim
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
import librosa

M_A, M_B = 100, 30
BLOCK = 10
CASES = [("office_1", 33, 11), ("frl_apartment_5", 186, 121), ("room_0", 43, 121)]
OUT_NPZ = Path(str(MEAS_OUT / "warmup_specs.npz"))
OUT_PNG = Path(__file__).resolve().parents[2] / "outputs/diagnose_rlr_noise_out/warmup_simulator.png"

chirp, _ = librosa.load(str(CHIRP_PATH), sr=spectrogram.SAMPLE_RATE, mono=True)
MC = str(MATERIAL_CONFIG)


def build(scene):
    class A:
        pass
    a = A()
    a.scene = str(scene_mesh(scene))
    a.sensor_height = SENSOR_HEIGHT
    a.material_config = MC
    a.out_dir = str(MEAS_OUT / "warmup_out")
    a.indirect_ray_count = INDIRECT_RAY_COUNT
    a.thread_count = THREAD_COUNT
    a.gpu_device_id = 0
    return audio.build_simulator(a)


def sigma_var(specs):
    """sqrt(srednia po komorkach z wariancji po renderach) = szum 1 renderu."""
    return float(np.sqrt(np.var(specs.astype(np.float64), axis=0, ddof=1).mean()))


def render(sim, pos, mc):
    """Sciezka PRODUKCYJNA: run_simulation=False (jedna symulacja audio)."""
    obs, _, _ = audio.phase3_echolocation(sim, pos, 0.0, mc, run_simulation=False)
    _e, s = spectrogram.render_spectrogram(np.transpose(np.array(obs["audio_sensor"])), chirp)
    return s.astype(np.float32)


store, n_sims = {}, 0
results = {}
for scene, locA, locB in CASES:
    positions = load_scene_locations(scene)[1]
    sim = build(scene)
    n_sims += 1
    print(f"\n[{n_sims}/3 Simulator] {scene}: pozycja A={locA} ({M_A} renderow), "
          f"potem B={locB} ({M_B}) w TEJ SAMEJ instancji", flush=True)
    try:
        mc = MC
        specA, specB, times = [], [], []
        for i in range(M_A):
            t0 = time.perf_counter()
            specA.append(render(sim, positions[locA], mc))
            times.append(time.perf_counter() - t0)
            mc = None
        for i in range(M_B):
            specB.append(render(sim, positions[locB], None))
        specA, specB = np.stack(specA), np.stack(specB)
    finally:
        sim.close()

    store[f"{scene}|A"] = specA
    store[f"{scene}|B"] = specB
    eA = specA.mean(axis=(1, 2, 3))
    eB = specB.mean(axis=(1, 2, 3))
    sA = [sigma_var(specA[i:i + BLOCK]) for i in range(0, M_A, BLOCK)]
    sB = [sigma_var(specB[i:i + BLOCK]) for i in range(0, M_B, BLOCK)]
    results[scene] = dict(locA=locA, locB=locB, eA=eA, eB=eB, sA=np.array(sA),
                          sB=np.array(sB), t_med=float(np.median(times)))
    print(f"  sigma per blok 10, pozycja A: " + " ".join(f"{v:.5f}" for v in sA))
    print(f"  sigma per blok 10, pozycja B: " + " ".join(f"{v:.5f}" for v in sB))
    print(f"  mediana czasu renderu: {np.median(times)*1000:.1f} ms")

np.savez_compressed(OUT_NPZ, **store)
print(f"\n  Surowe spektrogramy -> {OUT_NPZ} ({OUT_NPZ.stat().st_size/2**20:.0f} MiB)")

# --- analiza ---------------------------------------------------------------
print(f"\n{'='*100}\n  WERDYKT\n{'='*100}")
for scene, r in results.items():
    sA, sB = r["sA"], r["sB"]
    # Stan ustalony i jego niepewnosc: bloki od 5. wzwyz (rendery 40+)
    plateau = sA[4:]
    plat_mean, plat_sd = float(plateau.mean()), float(plateau.std(ddof=1))
    print(f"\n  --- {scene} (A={r['locA']}, B={r['locB']}) ---")
    print(f"  stan ustalony (bloki 5-10, rendery 40-99): {plat_mean:.5f} "
          f"+- {plat_sd:.5f} ({100*plat_sd/plat_mean:.1f} % SD bloku)")
    for i, v in enumerate(sA):
        dev = (v - plat_mean) / plat_mean * 100
        z = (v - plat_mean) / plat_sd
        flag = "  <-- powyzej 2 SD" if z > 2 else ""
        print(f"    blok {i+1:>2} (r{i*BLOCK:>3}-{i*BLOCK+BLOCK-1:>3}): {v:.5f}  "
              f"{dev:+6.1f} %  {z:+5.2f} SD{flag}")
    # energia: dryf sredniej?
    eA = r["eA"]
    early, late = eA[:10], eA[40:]
    se = float(np.sqrt(early.var(ddof=1)/len(early) + late.var(ddof=1)/len(late)))
    de = float(early.mean() - late.mean())
    print(f"  ENERGIA  pierwsze 10: {early.mean():.6f}   rendery 40+: {late.mean():.6f}   "
          f"roznica {de:+.3e} = {abs(de)/se:.2f} SE ({de/late.mean()*100:+.3f} %)")
    # per pozycja?
    print(f"  POZYCJA B (ta sama instancja, po {M_A} renderach):")
    for i, v in enumerate(sB):
        z = (v - sB[1:].mean()) / plat_sd if len(sB) > 1 else 0
        print(f"    blok {i+1} (r{i*BLOCK}-{i*BLOCK+BLOCK-1}): {v:.5f}   "
              f"odchylenie od pozostalych blokow B: {z:+.2f} SD(A)")
    b_first, b_rest = sB[0], sB[1:].mean()
    print(f"    pierwszy blok B / pozostale B = {b_first/b_rest:.3f}   "
          f"(dla porownania na pozycji A: {sA[0]/plat_mean:.3f})")

# --- wykres ----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for j, (scene, r) in enumerate(results.items()):
    ax = axes[0, j]
    ax.plot(np.arange(M_A), r["eA"], lw=0.8, color="tab:blue", label=f"poz. A (id={r['locA']})")
    ax.plot(np.arange(M_A, M_A + M_B), r["eB"], lw=0.8, color="tab:orange",
            label=f"poz. B (id={r['locB']})")
    ax.axvline(M_A, color="k", ls=":", lw=1)
    ax.set_title(f"{scene} — energia spektrogramu", fontsize=10)
    ax.set_xlabel("numer renderu w instancji")
    ax.set_ylabel("średnia |STFT|")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, j]
    xa = np.arange(len(r["sA"])) * BLOCK + BLOCK / 2
    xb = M_A + np.arange(len(r["sB"])) * BLOCK + BLOCK / 2
    plat = r["sA"][4:]
    ax.axhspan(plat.mean() - plat.std(ddof=1), plat.mean() + plat.std(ddof=1),
               color="tab:green", alpha=0.15, label="stan ustalony ±1 SD")
    ax.axhline(plat.mean(), color="tab:green", lw=1)
    ax.plot(xa, r["sA"], "o-", color="tab:blue", ms=4, label="poz. A")
    ax.plot(xb, r["sB"], "s-", color="tab:orange", ms=4, label="poz. B (ta sama instancja)")
    ax.axvline(M_A, color="k", ls=":", lw=1)
    ax.set_title(f"{scene} — szum 1 renderu (estymator wariancyjny, bloki po {BLOCK})",
                 fontsize=9)
    ax.set_xlabel("numer renderu w instancji")
    ax.set_ylabel("sigma_1")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Rozgrzewka Simulatora: czy per konstrukcja, czy per pozycja?  "
             "(ścieżka produkcyjna, 1 symulacja audio/render)", fontsize=11)
fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=130)
print(f"\n  Wykres -> {OUT_PNG}")
print(f"  Konstrukcji Simulatora: {n_sims} (prog ~30)")
