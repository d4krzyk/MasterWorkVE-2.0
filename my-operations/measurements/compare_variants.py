#!/usr/bin/env python3
"""KONTROLA DATASETU: czy warianty `main` i `patched` zgadzaja sie tam, gdzie musza.

Pytanie: oba warianty maja te same lokalizacje, te same katy i te sama konfiguracje
kamery — roznic ma je WYLACZNIE geometria sceny. Ten skrypt sprawdza, czy tak jest
w wygenerowanych plikach, i mierzy, jak duza jest faktyczna roznica w danych.

CO MUSI BYC IDENTYCZNE (inaczej porownanie 36 vs 4 miedzy wariantami jest niewazne):
  * zbior i kolejnosc probek: `location_id`, `angle_deg`, `position`,
  * konfiguracja zapisana w atrybutach: wysokosc sluchacza i kamery, HFOV,
    rozdzielczosc, liczba promieni, potok spektrogramu, suma kontrolna chirpa.

CO MA SIE ROZNIC I DLACZEGO:
  * `echo` — bo domkniecie sceny zmienia pole akustyczne (o to chodzi w eksperymencie),
  * `rgb` i `depth` — bo dolozony sufit JEST widoczny dla kamery. To nie jest efekt
    uboczny do przemilczenia: wariant `patched` nie odtwarza juz obrazu z VisualEchoes,
    wiec porownania z praca zrodlowa wolno robic TYLKO na wariancie `main`.
  * `sigma_1_probe` i `n_planned` — wiecej odbic stochastycznych to wyzszy szum Monte
    Carlo, wiec regula adaptacyjna zada wiecej renderow. To poprawna reakcja, nie blad.

Kontrola jakosci obu wariantow osobno: udzial probek ponizej TARGET_SNR, probki
obciete przez N_MAX, NaN-y, probki niezapisane.

Uruchomienie (bez GPU):
    python my-operations/measurements/compare_variants.py
    python my-operations/measurements/compare_variants.py --samples 400
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_core import paths as P
from echo_core.params import N_MAX, TARGET_SNR
from echo_core.scenes import SCENE_ORDER

# Atrybuty, ktore MUSZA byc identyczne w obu wariantach. Nie ma tu `scene_id`
# ani `variant` — te z definicji sie roznia i to jest ich cala rola.
MUST_MATCH = [
    "listener_height", "camera_height", "camera_hfov_deg", "camera_resolution",
    "indirect_ray_count", "thread_count", "sample_rate", "channel_count",
    "spectrogram_shape", "echo_ms", "echo_samples", "stft_n_fft", "stft_win_length",
    "chirp_sha256", "material_config_sha256", "averaging_domain", "n_angles",
    "enable_materials", "load_semantic_mesh",
]


def h5(path):
    import h5py
    return h5py.File(path, "r")


def scene_paths(scene):
    P.set_variant("main")
    a = P.scene_h5(scene)
    P.set_variant("patched")
    b = P.scene_h5(scene)
    P.set_variant("main")
    return a, b


def quality(f, label):
    """Kontrola jakosci jednego pliku -> slownik liczb."""
    w = f["written"][:].astype(bool)
    snr = f["snr_final"][:]
    n_tot = f["n_total"][:]
    clamped = f["clamped"][:]
    below = (snr[w] < TARGET_SNR)
    at_max = (n_tot[w] >= N_MAX)
    return {
        "label": label,
        "probek": int(w.size),
        "zapisanych": int(w.sum()),
        "ponizej_SNR": int(below.sum()),
        "przy_N_MAX": int(at_max.sum()),
        "przy_N_MAX_i_ponizej_SNR": int((below & at_max).sum()),
        "clamped_max": int((clamped[w] == b"max").sum()),
        "sigma1_med": float(np.median(f["locations"]["sigma_1_probe"][:])),
        "N_med": float(np.median(f["locations"]["n_planned"][:])),
        "snr_med": float(np.median(snr[w])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=250,
                    help="ile probek na scene losowac do ciezkich porownan (echo/rgb/depth)")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    P.set_variant("patched")
    shared = [s for s in SCENE_ORDER if P.has_patch(s)]
    P.set_variant("main")

    rng = np.random.default_rng(args.seed)
    rows, qual_main, qual_patch = [], [], []
    problems = []

    print(f"\n  Scen wspolnych dla obu wariantow: {len(shared)}")
    print(f"  {'scena':<18}{'probek':>8}{'siatka':>9}{'atrybuty':>10}"
          f"{'echo !=':>9}{'rgb !=':>8}{'depth !=':>9}{'sigma1 x':>10}{'N x':>7}")
    print("  " + "-" * 90)

    for scene in shared:
        pa, pb = scene_paths(scene)
        if not (pa.exists() and pb.exists()):
            problems.append(f"{scene}: brak pliku w jednym z wariantow")
            continue
        with h5(pa) as A, h5(pb) as B:
            # --- 1. siatka probkowania musi byc identyczna --------------------
            same_grid = True
            for k in ("location_id", "angle_deg", "position"):
                if not np.array_equal(A[k][:], B[k][:]):
                    same_grid = False
                    problems.append(f"{scene}: rozna tablica `{k}` miedzy wariantami")
            # --- 2. atrybuty konfiguracji ------------------------------------
            bad_attr = []
            for k in MUST_MATCH:
                va, vb = A.attrs.get(k), B.attrs.get(k)
                same = (np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb)
                if not same:
                    bad_attr.append(k)
                    problems.append(f"{scene}: atrybut `{k}` rozny: {va!r} vs {vb!r}")
            # kontrola pozytywna: wariant MUSI byc zapisany i musi sie roznic
            if A.attrs.get("variant") != "main" or B.attrs.get("variant") != "patched":
                problems.append(f"{scene}: zly atrybut `variant` "
                                f"({A.attrs.get('variant')!r}/{B.attrs.get('variant')!r})")

            # --- 3. ile faktycznie roznia sie dane ---------------------------
            w = (A["written"][:].astype(bool) & B["written"][:].astype(bool))
            idx = np.flatnonzero(w)
            take = np.sort(rng.choice(idx, size=min(args.samples, idx.size), replace=False))
            ea = A["echo"][take].astype(np.float32)
            eb = B["echo"][take].astype(np.float32)
            ra, rb = A["rgb"][take], B["rgb"][take]
            da, db = A["depth"][take], B["depth"][take]
            echo_diff = float(np.mean(np.abs(ea - eb) > 1e-3) * 100)
            rgb_diff = float(np.mean(ra != rb) * 100)
            depth_diff = float(np.mean(np.abs(da - db) > 1e-3) * 100)
            # energia echa: domkniecie ma ja PODNIESC
            e_ratio = float(eb.mean() / ea.mean())

            s1a = np.median(A["locations"]["sigma_1_probe"][:])
            s1b = np.median(B["locations"]["sigma_1_probe"][:])
            na = np.median(A["locations"]["n_planned"][:])
            nb = np.median(B["locations"]["n_planned"][:])

            qual_main.append(quality(A, scene))
            qual_patch.append(quality(B, scene))
            rows.append({"scene": scene, "echo_diff_pct": echo_diff, "rgb_diff_pct": rgb_diff,
                         "depth_diff_pct": depth_diff, "echo_energy_ratio": e_ratio,
                         "sigma1_ratio": float(s1b / s1a), "n_ratio": float(nb / na),
                         "grid_identical": same_grid, "attrs_identical": not bad_attr})

            print(f"  {scene:<18}{int(w.sum()):>8}"
                  f"{('OK' if same_grid else 'BLAD'):>9}{('OK' if not bad_attr else 'BLAD'):>10}"
                  f"{echo_diff:>8.1f}%{rgb_diff:>7.1f}%{depth_diff:>8.1f}%"
                  f"{s1b/s1a:>9.2f}x{nb/na:>6.2f}x")

    # --- podsumowanie ------------------------------------------------------
    print("  " + "-" * 90)
    ok_grid = all(r["grid_identical"] for r in rows)
    ok_attr = all(r["attrs_identical"] for r in rows)
    print(f"  siatka probkowania identyczna we wszystkich scenach: "
          f"{'TAK' if ok_grid else 'NIE'}")
    print(f"  konfiguracja (atrybuty) identyczna:                  "
          f"{'TAK' if ok_attr else 'NIE'}")
    if rows:
        print(f"  echo rozne w {np.median([r['echo_diff_pct'] for r in rows]):.1f} % komorek "
              f"(mediana), energia patched/main "
              f"{np.median([r['echo_energy_ratio'] for r in rows]):.2f}x")
        print(f"  rgb rozne w {np.median([r['rgb_diff_pct'] for r in rows]):.1f} %, "
              f"depth w {np.median([r['depth_diff_pct'] for r in rows]):.1f} % pikseli")
        print(f"  sigma_1 patched/main {np.median([r['sigma1_ratio'] for r in rows]):.2f}x, "
              f"N {np.median([r['n_ratio'] for r in rows]):.2f}x")

    for name, q in (("main", qual_main), ("patched", qual_patch)):
        tot = sum(x["probek"] for x in q)
        wr = sum(x["zapisanych"] for x in q)
        bad = sum(x["przy_N_MAX_i_ponizej_SNR"] for x in q)
        below = sum(x["ponizej_SNR"] for x in q)
        print(f"\n  [{name}] {len(q)} scen · probek {wr}/{tot} "
              f"({'KOMPLET' if wr == tot else 'NIEKOMPLETNE'})")
        print(f"    ponizej SNR {TARGET_SNR}: {below} ({below/max(wr,1)*100:.3f} %)"
              f" · z tego przy N_MAX={N_MAX}: {bad} ({bad/max(wr,1)*100:.3f} %)")

    if problems:
        print("\n  PROBLEMY:")
        for p in problems[:20]:
            print(f"    - {p}")
    else:
        print("\n  Brak niezgodnosci tam, gdzie warianty musza byc identyczne.")

    out = P.REPO_ROOT / "outputs/measurements/dataset_check/variant_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shared_scenes": shared, "per_scene": rows,
                               "quality_main": qual_main, "quality_patched": qual_patch,
                               "problems": problems, "samples_per_scene": args.samples},
                              indent=2, ensure_ascii=False))
    print(f"\n  zapisano: {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
