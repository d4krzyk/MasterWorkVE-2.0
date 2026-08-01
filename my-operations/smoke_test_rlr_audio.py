#!/usr/bin/env python3
"""Smoke test potoku RLRAudioPropagation w tym buildzie habitat-sim.

Sprawdza po kolei: API audio, budowe Simulatora, pozycje na navmeshu, echolokacje,
poprawnosc RIR, spektrogram, zapis artefaktow i porownanie katow. Kazda faza jest
niezalezna i raportuje PASS/FAIL/SKIP.

To NIE jest pytest — to recznie uruchamiany skrypt kontrolny:

    conda activate habitat
    python my-operations/smoke_test_rlr_audio.py \
        --scene sound-spaces/data/scene_datasets/replica/room_0/habitat/mesh_semantic.ply \
        --chirp my-operations/sweep_audio/3ms_sweep.wav

Wlasciwy potok (budowa Simulatora, echolokacja, spektrogram) mieszka w echo_core/
i jest STAD importowany, nie duplikowany — to ten sam kod, ktorego uzywa generator
i diagnostyka, wiec smoke test faktycznie testuje produkcje.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # brak X displaya w tym srodowisku (headless smoke test)

import matplotlib.pyplot as plt
import numpy as np
import quaternion  # noqa: F401
import habitat_sim
import soundfile as sf

from echo_core.audio import PhaseFailure, build_simulator, phase3_echolocation
from echo_core.spectrogram import (ECHO_SAMPLES, EXPECTED_SPEC_SHAPE, N_CHANNELS, SAMPLE_RATE,
                                   STFT_HOP_LENGTH, STFT_N_FFT, render_spectrogram)

BONUS_ANGLES = (0.0, 10.0, 90.0)
NAV_SEED = 42  # ustalony seed pathfindera dla powtarzalnosci wyniku


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene",
        required=True,
        help="Sciezka do <scena>/habitat/mesh_semantic.ply (Replica). "
        "Navmesh (mesh_semantic.navmesh) musi lezec obok, pod ta sama nazwa "
        "bazowa - habitat-sim wykrywa go automatycznie.",
    )
    p.add_argument("--angle", type=float, default=0.0, help="Azymut agenta w stopniach (domyslnie 0).")
    p.add_argument(
        "--material-config",
        default=None,
        help="Opcjonalna sciezka do JSON z materialami akustycznymi "
        "(wlacza acousticsConfig.enableMaterials + semantic mesh).",
    )
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "outputs/rlr_test_out"),
                   help="Katalog na artefakty (domyslnie <repo>/outputs/rlr_test_out).")
    p.add_argument(
        "--chirp",
        required=True,
        help="Sciezka do sygnalu wzbudzajacego (np. 3ms_sweep.wav) uzywanego "
        "do konwolucji z RIR w fazie 5.",
    )
    return p.parse_args()


def run_phase(results, name, fn, *fargs, required=True, **fkwargs):
    """Uruchamia jedna faze w try/except, zapisujac PASS/FAIL/SKIP w results.

    Jesli jakas wczesniejsza wymagana faza juz padla, kolejne fazy zalezne od
    jej wynikow sa oznaczane SKIP zamiast probowac dzialac na niepelnym
    stanie (np. faza 3 bez dzialajacego symulatora z fazy 1).
    """
    if any(status == "FAIL" for status, _ in results.values()):
        results[name] = ("SKIP", "pominieto - wczesniejsza faza nie powiodla sie")
        print(f"\n=== {name}: SKIP (wczesniejsza faza padla) ===")
        return None

    print(f"\n=== {name} ===")
    try:
        out = fn(*fargs, **fkwargs)
        results[name] = ("PASS", "")
        return out
    except Exception as e:  # celowo szerokie - to jest smoke test, chcemy
        # zobaczyc KAZDY blad z pelnym tracebackiem, nie tylko oczekiwane typy
        msg = f"{type(e).__name__}: {e}"
        results[name] = ("FAIL", msg)
        print(f"[FAIL] {name}: {msg}", file=sys.stderr)
        traceback.print_exc()
        return None


def phase0_introspect():
    print("habitat_sim.__version__:", getattr(habitat_sim, "__version__", "<brak atrybutu>"))

    if not hasattr(habitat_sim, "AudioSensorSpec"):
        raise PhaseFailure(
            "habitat_sim.AudioSensorSpec nie istnieje - ten build habitat-sim "
            "NIE ma wkompilowanego wsparcia audio. Przebuduj z flaga --audio "
            "(patrz habitat-sim/README.md, budowanie z flaga --audio) i upewnij sie, "
            "ze BUILD_WITH_AUDIO=ON w CMakeCache.txt."
        )
    print("habitat_sim.AudioSensorSpec: dostepny")

    spec = habitat_sim.AudioSensorSpec()
    print("dir(AudioSensorSpec()):")
    print(" ", [a for a in dir(spec) if not a.startswith("_")])
    print("dir(AudioSensorSpec().acousticsConfig):")
    print(" ", [a for a in dir(spec.acousticsConfig) if not a.startswith("_")])
    print("dir(AudioSensorSpec().channelLayout):")
    print(" ", [a for a in dir(spec.channelLayout) if not a.startswith("_")])

    if not hasattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType"):
        raise PhaseFailure("habitat_sim.sensor.RLRAudioPropagationChannelLayoutType nie istnieje.")
    layout_types = [a for a in dir(habitat_sim.sensor.RLRAudioPropagationChannelLayoutType) if not a.startswith("_")]
    print("RLRAudioPropagationChannelLayoutType warianty:", layout_types)
    if "Binaural" not in layout_types:
        raise PhaseFailure("Wariant 'Binaural' nie jest dostepny w tej wersji API channelLayout.")


def phase2_position(sim):
    if not sim.pathfinder.is_loaded:
        raise PhaseFailure(
            "Navmesh nie zostal wczytany (pathfinder.is_loaded == False). "
            "Dla Replica habitat-sim szuka pliku <baza>.navmesh obok podanej "
            "sceny - sprawdz, czy mesh_semantic.navmesh lezy w tym samym "
            "katalogu co mesh_semantic.ply."
        )
    sim.pathfinder.seed(NAV_SEED)
    point = sim.pathfinder.get_random_navigable_point()
    print("Losowy punkt nawigowalny (seed =", NAV_SEED, "):", point)
    return point


def phase4_validate_rir(obs):
    raw = np.array(obs["audio_sensor"])  # (kanaly, próbki), patrz getObservationSpace w AudioSensor.cpp
    print("Surowy ksztalt obs['audio_sensor']:", raw.shape, raw.dtype)

    rir = np.transpose(raw)  # (n_samples, 2) - to jest uklad, ktorego oczekujemy dalej
    print("Ksztalt po transpozycji:", rir.shape)

    if np.isnan(rir).any() or np.isinf(rir).any():
        raise PhaseFailure("RIR zawiera NaN/Inf - symulacja akustyczna jest niestabilna dla tej konfiguracji.")

    metrics = {"n_samples": int(rir.shape[0]), "n_channels": int(rir.shape[1]), "channels": []}
    for ch in range(rir.shape[1]):
        col = rir[:, ch]
        stats = {
            "dtype": str(col.dtype),
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "rms": float(np.sqrt(np.mean(col**2))),
        }
        print(f"  kanal {ch}: {stats}")
        metrics["channels"].append(stats)

    if not np.any(rir):
        raise PhaseFailure(
            "RIR to same zera. Przy zerowej odleglosci zrodlo-odbiornik "
            "ray-tracer bywa niestabilny (zdegenerowany promien zerowej "
            "dlugosci) - sprobuj przesunac zrodlo o ok. 5 cm zamiast dokladnej "
            "wspollokacji ze sluchaczem."
        )

    # Proxy na "direct path": pierwsza probka, w ktorej OBA kanaly razem
    # przekraczaja 1% globalnego maksimum amplitudy. Dla echolokacji (zrodlo
    # = odbiornik) oczekujemy indeksu bliskiego zeru - dzwiek dociera do
    # sluchacza niemal natychmiast.
    combined = np.max(np.abs(rir), axis=1)
    threshold = 0.01 * combined.max()
    direct_path_idx = int(np.argmax(combined > threshold))
    metrics["direct_path_sample_idx"] = direct_path_idx
    metrics["direct_path_time_ms"] = direct_path_idx / SAMPLE_RATE * 1000.0
    metrics["rir_length_samples"] = int(rir.shape[0])
    metrics["rir_length_seconds"] = rir.shape[0] / SAMPLE_RATE

    print(f"  proxy direct path: probka {direct_path_idx} ({metrics['direct_path_time_ms']:.3f} ms)")
    print(f"  dlugosc RIR: {rir.shape[0]} probek ({metrics['rir_length_seconds']:.4f} s)")

    return rir, metrics


def phase5_spectrogram(rir, chirp_path):
    chirp_file = Path(chirp_path)
    if not chirp_file.exists():
        raise PhaseFailure(f"Plik chirpu nie istnieje: {chirp_file}")

    import librosa

    chirp, loaded_sr = librosa.load(str(chirp_file), sr=SAMPLE_RATE, mono=True)
    print(f"Chirp: {chirp.shape[0]} probek @ {loaded_sr} Hz")

    echo, spec = render_spectrogram(rir, chirp)
    print("Ksztalt echa (60ms):", echo.shape)
    print("Ksztalt spektrogramu:", spec.shape)

    if spec.shape != EXPECTED_SPEC_SHAPE:
        # Zdiagnozuj, ktory parametr prawdopodobnie sie nie zgadza, zamiast
        # samego surowego AssertionError.
        _, expected_f, expected_t = EXPECTED_SPEC_SHAPE
        got_c, got_f, got_t = spec.shape
        hint = []
        if got_c != N_CHANNELS:
            hint.append(f"liczba kanalow {got_c} != {N_CHANNELS} - sprawdz channelLayout.channelCount")
        if got_f != expected_f:
            hint.append(f"osie czestotliwosci {got_f} != {expected_f} - sprawdz n_fft (oczekiwane {STFT_N_FFT})")
        if got_t != expected_t:
            hint.append(
                f"liczba ramek {got_t} != {expected_t} - sprawdz hop_length "
                f"(oczekiwane {STFT_HOP_LENGTH}) albo dlugosc echa (oczekiwane {ECHO_SAMPLES} probek)"
            )
        raise PhaseFailure(
            f"Ksztalt spektrogramu {spec.shape} != oczekiwany {EXPECTED_SPEC_SHAPE}. " + "; ".join(hint)
        )

    return echo, spec, chirp


def phase6_artifacts(out_dir, rir, echo, spec, obs, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    sf.write(out_dir / "rir.wav", rir, SAMPLE_RATE)
    sf.write(out_dir / "echo_60ms.wav", echo.T, SAMPLE_RATE)

    # --- wykres czasowy RIR ---
    t_ms = np.arange(rir.shape[0]) / SAMPLE_RATE * 1000.0
    fig, axes = plt.subplots(rir.shape[1], 1, figsize=(10, 6), sharex=True)
    for ch in range(rir.shape[1]):
        axes[ch].plot(t_ms, rir[:, ch], linewidth=0.5)
        axes[ch].set_ylabel(f"kanal {ch}")
    axes[-1].set_xlabel("czas [ms]")
    fig.suptitle("RIR (odpowiedz impulsowa)")
    fig.tight_layout()
    fig.savefig(out_dir / "rir_waveform.png", dpi=150)
    plt.close(fig)

    # --- spektrogram ---
    import librosa

    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=STFT_N_FFT)
    fig, axes = plt.subplots(1, spec.shape[0], figsize=(10, 4))
    if spec.shape[0] == 1:
        axes = [axes]
    for ch in range(spec.shape[0]):
        im = axes[ch].imshow(
            spec[ch],
            origin="lower",
            aspect="auto",
            extent=[0, spec.shape[2], freqs[0], freqs[-1]],
        )
        axes[ch].set_title(f"kanal {ch}")
        axes[ch].set_xlabel("ramka")
        axes[ch].set_ylabel("czestotliwosc [Hz]")
        fig.colorbar(im, ax=axes[ch])
    fig.suptitle("Spektrogram magnitudy echa (60ms)")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrogram.png", dpi=150)
    plt.close(fig)

    # --- RGB / depth sanity check ---
    plt.imsave(out_dir / "rgb.png", obs["rgb"][:, :, :3])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(obs["depth"], cmap="viridis")
    fig.colorbar(im, ax=ax, label="glebia [m]")
    ax.set_title("Depth")
    fig.savefig(out_dir / "depth.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Artefakty zapisane w:", out_dir)


def phase7_angle_comparison(sim, position, material_config, chirp):
    specs = {}
    for angle in BONUS_ANGLES:
        obs, _, _ = phase3_echolocation(sim, position, angle, material_config)
        raw = np.array(obs["audio_sensor"])
        rir = np.transpose(raw)
        if not np.any(rir):
            raise PhaseFailure(f"RIR dla kata {angle} to same zera - nie mozna policzyc porownania.")
        _, spec = render_spectrogram(rir, chirp)
        specs[angle] = spec

    n = len(BONUS_ANGLES)
    rmse = np.zeros((n, n))
    for i, a in enumerate(BONUS_ANGLES):
        for j, b in enumerate(BONUS_ANGLES):
            rmse[i, j] = np.sqrt(np.mean((specs[a] - specs[b]) ** 2))

    print("Macierz RMSE miedzy spektrogramami (kolejnosc katow:", BONUS_ANGLES, "):")
    print(rmse)

    idx_0, idx_10, idx_90 = 0, 1, 2
    rmse_0_10 = rmse[idx_0, idx_10]
    rmse_0_90 = rmse[idx_0, idx_90]
    print(f"RMSE(0, 10) = {rmse_0_10:.6g}, RMSE(0, 90) = {rmse_0_90:.6g}")

    # Heurystyka: jesli zmiana o 10 stopni daje ponizej 5% roznicy, jaka daje
    # zmiana o 90 stopni, sygnal roznicujacy sasiednie katy moze byc zagluszony
    # przez stochastyczny szum ray-tracera (np. za mala liczba promieni
    # indirect), a nie realny brak informacji przestrzennej.
    warning = None
    if rmse_0_90 > 0 and rmse_0_10 < 0.05 * rmse_0_90:
        warning = (
            "RMSE(0,10) jest ponizej 5% RMSE(0,90) - roznica miedzy sasiednimi "
            "katami co 10 stopni jest bliska szumowi. Rozwaz zwiekszenie "
            "acousticsConfig.indirectRayCount (obecnie 500) przed wygenerowaniem "
            "pelnego zbioru 36 orientacji x 18 scen."
        )
        print("UWAGA (problem metodologiczny):", warning)

    return {"angles": list(BONUS_ANGLES), "rmse_matrix": rmse.tolist(), "warning": warning}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    results = {}
    run_phase(results, "Faza 0: introspekcja API", phase0_introspect)

    sim = run_phase(results, "Faza 1: konfiguracja symulatora", build_simulator, args)

    position = None
    if sim is not None:
        position = run_phase(results, "Faza 2: pozycja", phase2_position, sim)

    obs = listener_pos = None
    if sim is not None and position is not None:
        out = run_phase(
            results,
            "Faza 3: echolokacja",
            phase3_echolocation,
            sim,
            position,
            args.angle,
            args.material_config,
        )
        if out is not None:
            obs, listener_pos, _ = out

    rir = metrics = None
    if obs is not None:
        out = run_phase(results, "Faza 4: walidacja RIR", phase4_validate_rir, obs)
        if out is not None:
            rir, metrics = out

    echo = spec = chirp = None
    if rir is not None:
        out = run_phase(results, "Faza 5: spektrogram", phase5_spectrogram, rir, args.chirp)
        if out is not None:
            echo, spec, chirp = out
            metrics["spectrogram_shape"] = list(spec.shape)

    if rir is not None and echo is not None and metrics is not None:
        run_phase(results, "Faza 6: artefakty", phase6_artifacts, out_dir, rir, echo, spec, obs, metrics)

    if sim is not None and position is not None and chirp is not None:
        bonus = run_phase(
            results,
            "Faza 7: porownanie katow (bonus)",
            phase7_angle_comparison,
            sim,
            position,
            args.material_config,
            chirp,
        )
        if bonus is not None and metrics is not None:
            metrics["angle_comparison"] = bonus
            with open(out_dir / "report.json", "w") as f:
                json.dump(metrics, f, indent=2)

    if sim is not None:
        sim.close()

    print("\n=== PODSUMOWANIE ===")
    overall_ok = True
    for name, (status, msg) in results.items():
        print(f"  [{status}] {name}" + (f" - {msg}" if msg else ""))
        if status == "FAIL":
            overall_ok = False
    print("WERDYKT:", "PASS" if overall_ok else "FAIL")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
