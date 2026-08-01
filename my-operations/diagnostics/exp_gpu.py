"""Skalowanie zuzycia pamieci GPU/RSS przy dlugich seriach renderow."""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Kolejnosc importow (quaternion przed habitat_sim) wymagana przez lokalny patch
# tego repo - patrz habitat-sim/local_changes.patch.
import quaternion  # noqa: F401
import habitat_sim

from echo_core.audio import build_simulator, phase3_echolocation
from echo_core.spectrogram import (ECHO_MS, ECHO_SAMPLES, EXPECTED_SPEC_SHAPE, SAMPLE_RATE,
                                   STFT_HOP_LENGTH, STFT_N_FFT, STFT_WIN_LENGTH,
                                   render_spectrogram)

from .common import (CHIRP_PATH, OUT_DIR, PRODUCTION_SENSOR_HEIGHT, REPLICA_MATERIAL_CONFIG, REPO_ROOT, _get_spec, _points_path, build_sim, load_point_position)


# --- BLOK 2: czy tysiace renderow w JEDNEJ instancji ciekna? -----------------
#
# Wiemy, ze ~30 KONSTRUKCJI Simulatora w jednym procesie kladzie karte sprzetowo
# (wyciek GL/EGL na sim.close(); odzysk wymaga prawdziwego resetu PCI:
# `echo 1 > /sys/bus/pci/devices/<id>/reset` po wyladowaniu modulow nvidia*).
# To jednak inny wzorzec zuzycia zasobow niz tysiace RENDEROW w jednej instancji -
# a to wlasnie jest wzorzec produkcyjny ("jeden dlugo zyjacy Simulator na scene").
# Jesli render tez cieknie, architektura wymaga rotacji instancji w obrebie sceny
# (co jest bezpieczne dzieki werdyktowi BEZPIECZNY z e1_checkpoint_boundary_merge).
#
# Mierzymy trzy rzeczy naraz, bo wyciek moze sie ujawnic w kazdej z osobna:
#  - pamiec GPU (karta),
#  - RSS procesu (host - alokacje po stronie CPU tez potrafia rosnac),
#  - czas renderu (czesto rosnie WCZESNIEJ niz sama pamiec, np. przy narastajacej
#    liczbie zywych obiektow do przejrzenia).

GPUMEM_SCENE = "room_0"

GPUMEM_RENDERS = 3000

GPUMEM_SAMPLE_EVERY = 100

GPUMEM_RAYS = 500

GPUMEM_THREADS = 1

def _gpu_mem_mib():
    """Pamiec GPU zajeta lacznie (MiB) - z nvidia-smi, bo habitat trzyma kontekst
    przez EGL/GL, a nie jako 'compute app', wiec per-proces czesto nie widac."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")

def _proc_rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")

def _largest_scene_render_count(n_renders_per_sample):
    """Ile renderow przypadnie na najwieksza scene - cel, do ktorego ekstrapolujemy."""
    biggest, best = None, -1
    for sd in sorted((REPO_ROOT / "my-operations/metadata/replica").iterdir()):
        pf = sd / "points.txt"
        if not pf.exists():
            continue
        n = sum(1 for _ in pf.open())
        if n > best:
            biggest, best = sd.name, n
    return biggest, best, best * 36 * n_renders_per_sample

def run_gpu_memory_scale():
    print("\n=== BLOK 2: pamiec GPU i RSS przy tysiacach runSimulation() ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    n_prod = int(os.environ.get("VE_N_RENDERS", "10"))  # N z Bloku 1
    big_scene, big_points, big_renders = _largest_scene_render_count(n_prod)
    print(f"  Najwieksza scena: {big_scene} ({big_points} lokalizacji) -> "
          f"{big_points} x 36 katow x N={n_prod} = {big_renders} renderow w jednej instancji")

    result = {"scene": GPUMEM_SCENE, "renders": GPUMEM_RENDERS, "sample_every": GPUMEM_SAMPLE_EVERY,
              "rays": GPUMEM_RAYS, "threads": GPUMEM_THREADS, "sensor_height": PRODUCTION_SENSOR_HEIGHT,
              "n_renders_per_sample_assumed": n_prod,
              "largest_scene": big_scene, "largest_scene_points": big_points,
              "largest_scene_renders": big_renders,
              "gpu_total_mib": None}

    points = pd.read_csv(_points_path(GPUMEM_SCENE), sep="\t", header=None, names=["id", "a", "b", "c"])
    n_points = len(points)
    baseline_gpu = _gpu_mem_mib()
    samples = []

    sim = build_sim(scene_name=GPUMEM_SCENE, indirect_ray_count=GPUMEM_RAYS, thread_count=GPUMEM_THREADS,
                    material_config=REPLICA_MATERIAL_CONFIG, sensor_height=PRODUCTION_SENSOR_HEIGHT)
    try:
        # Cache pozycji, zeby snap_point nie mieszal sie do pomiaru czasu renderu.
        pos_cache = {}
        for i in range(GPUMEM_RENDERS):
            # Wzorzec produkcyjny: przechodzimy po lokalizacjach i 36 katach, a nie
            # renderujemy w kolko jednego stanu - inny stan to inna praca silnika.
            pid = (i // 36) % n_points
            ang = float((i % 36) * 10)
            if pid not in pos_cache:
                pos_cache[pid] = load_point_position(sim, GPUMEM_SCENE, pid)
            t0 = time.perf_counter()
            _get_spec(sim, pos_cache[pid], ang, mc, chirp)
            dt = time.perf_counter() - t0
            if i % GPUMEM_SAMPLE_EVERY == 0 or i == GPUMEM_RENDERS - 1:
                samples.append({"render": i, "gpu_mib": _gpu_mem_mib(), "rss_mib": _proc_rss_mib(),
                                "s_per_render": dt})
                s = samples[-1]
                print(f"    render {i:5d}: GPU {s['gpu_mib']:.0f} MiB, RSS {s['rss_mib']:.0f} MiB, "
                      f"{s['s_per_render']:.3f} s", flush=True)
    finally:
        sim.close()

    result["baseline_gpu_mib"] = baseline_gpu
    result["samples"] = samples
    result["gpu_total_mib"] = 16303.0  # RTX 5070 Ti

    # --- analiza trendu -------------------------------------------------------
    # Pierwsze 5 probek (500 renderow) to faza rozgrzewki - alokacje buforow,
    # cache'e sceny. Trend liczymy DOPIERO po niej, inaczej rozgrzewka udaje wyciek.
    warm = [s for s in samples if s["render"] >= 500]
    x = np.array([s["render"] for s in warm], dtype=np.float64)
    trends = {}
    for key in ("gpu_mib", "rss_mib", "s_per_render"):
        y = np.array([s[key] for s in warm], dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float("nan")
        trends[key] = {"per_1000_renders": slope * 1000.0,
                       "first": float(y[0]), "last": float(y[-1]),
                       "min": float(y.min()), "max": float(y.max()),
                       "std": float(y.std())}
    result["trends"] = trends

    _plot_gpu_memory(samples, result)

    gpu_rate = trends["gpu_mib"]["per_1000_renders"]
    rss_rate = trends["rss_mib"]["per_1000_renders"]
    time_rate = trends["s_per_render"]["per_1000_renders"]
    headroom = result["gpu_total_mib"] - trends["gpu_mib"]["last"]
    # Prog: rozrzut samego pomiaru nvidia-smi to kilkanascie MiB (desktop tez
    # alokuje), wiec za wyciek uznajemy dopiero tempo istotnie ponad ten szum.
    gpu_leaks = gpu_rate > 3 * trends["gpu_mib"]["std"] and gpu_rate > 10.0
    rss_leaks = rss_rate > 3 * trends["rss_mib"]["std"] and rss_rate > 10.0
    time_grows = time_rate > 0.02 * trends["s_per_render"]["first"]

    print("\n--- WERDYKT BLOK 2 ---")
    print(f"  GPU: {trends['gpu_mib']['first']:.0f} -> {trends['gpu_mib']['last']:.0f} MiB "
          f"(tempo {gpu_rate:+.1f} MiB / 1000 renderow, rozrzut {trends['gpu_mib']['std']:.1f})")
    print(f"  RSS: {trends['rss_mib']['first']:.0f} -> {trends['rss_mib']['last']:.0f} MiB "
          f"(tempo {rss_rate:+.1f} MiB / 1000 renderow, rozrzut {trends['rss_mib']['std']:.1f})")
    print(f"  czas/render: {trends['s_per_render']['first']:.3f} -> {trends['s_per_render']['last']:.3f} s "
          f"(tempo {time_rate:+.4f} s / 1000 renderow)")
    if not gpu_leaks and not rss_leaks and not time_grows:
        verdict = ("STABILNA - po fazie rozgrzewki ani pamiec GPU, ani RSS procesu, ani czas renderu nie "
                   f"rosna w tempie odrozniamym od szumu pomiaru na {GPUMEM_RENDERS} renderach. "
                   "Architektura 'jeden Simulator na scene' jest bezpieczna bez rotacji instancji.")
    else:
        parts = []
        if gpu_leaks:
            n_to_oom = headroom / gpu_rate * 1000.0
            parts.append(f"GPU rosnie {gpu_rate:.1f} MiB/1000 renderow; przy zapasie {headroom:.0f} MiB "
                         f"pamieci zabraknie po ~{n_to_oom:.0f} renderach "
                         f"({'PRZED' if n_to_oom < big_renders else 'PO'} koncu najwiekszej sceny, "
                         f"ktora wymaga {big_renders})")
        if rss_leaks:
            parts.append(f"RSS rosnie {rss_rate:.1f} MiB/1000 renderow")
        if time_grows:
            parts.append(f"czas renderu rosnie {time_rate:+.4f} s/1000 renderow")
        verdict = "ROSNIE - " + "; ".join(parts) + "."
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result

def _plot_gpu_memory(samples, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [s["render"] for s in samples]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, key, label, color in ((axes[0], "gpu_mib", "pamiec GPU [MiB]", "tab:red"),
                                  (axes[1], "rss_mib", "RSS procesu [MiB]", "tab:blue"),
                                  (axes[2], "s_per_render", "czas renderu [s]", "tab:green")):
        y = [s[key] for s in samples]
        ax.plot(x, y, "o-", ms=3, lw=1.2, color=color)
        ax.axvline(500, color="k", ls=":", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{result['scene']}, {result['renders']} renderow w JEDNEJ instancji "
                      f"({result['rays']} promieni / {result['threads']} watek)\n"
                      "linia kropkowana = koniec fazy rozgrzewki (500 renderow)", fontsize=10)
    axes[2].set_xlabel("numer renderu")
    fig.tight_layout()
    path = OUT_DIR / "gpu_memory_scale.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    result["plot_path"] = str(path)
    print(f"\n  Wykres zapisany: {path}")
