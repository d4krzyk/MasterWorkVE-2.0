#!/usr/bin/env python3
"""EKSPERYMENT: czy domkniecie sceny sufitem zbliza RT60 do SoundSpaces 1.0?

Pytanie: `frl_apartment_2` nie ma sufitu w skanie; nasze RT60 @ 1 kHz wynosi 0.186 s,
a prekomputowane RIR-y SoundSpaces 1.0 dla tej samej sceny daja 0.463 s (2.5x wiecej).
Na scenie ZAMKNIETEJ (`office_1`) oba silniki zgadzaja sie w ~10 % (0.358 vs 0.396 s).
Hipoteza: potok SS 1.0 domykal objetosc przed symulacja. Jesli jest prawdziwa, to
doklejenie sufitu do naszej siatki powinno PODNIESC nasze RT60 w kierunku 0.463 s.

To jest test falsyfikowalny w obie strony:
  * RT60 rosnie do ~0.4-0.5 s  -> hipoteza o domykaniu objetosci przez SS 1.0 zyskuje
    mocne wsparcie, a roznica wobec baseline'ow ma znane, usuwalne zrodlo;
  * RT60 rosnie slabo albo wcale -> brak sufitu NIE tlumaczy rozbieznosci i trzeba
    szukac gdzie indziej (inny model propagacji, inne materialy, inne przetwarzanie
    RIR-ow). Wtedy "naprawa" sceny niczego by nie naprawila.

METODA — jedyna zmienna to siatka:
  * te same pary (zrodlo, odbiornik), te same pozycje produkcyjne, ten sam silnik,
    ten sam estymator RT60 co zastosowany do danych SS 1.0 (soundspaces1_rt60.t60:
    Schroeder -> T20 w oknie -5..-25 dB -> RT60), te same pasma oktawowe;
  * ZRODLO ODDALONE o 1-3 m, nie wspollokowane. Powod: w echolokacji zrodlo siedzi
    w punkcie odbioru, przez co zanik jest wielonachyleniowy i okno -5..-25 dB trafia
    w inna czesc krzywej niz w RIR-ach SS 1.0. Ten sam przedzial odleglosci (1-3 m)
    filtruje pary w soundspaces1_rt60.py, wiec geometria jest dopasowana;
  * usrednianie po renderach w domenie ENERGII, po przefiltrowaniu KAZDEGO renderu
    osobno w domenie cisnienia. Usrednianie cisnienia kasowaloby losowe fazy odbic
    stochastycznych i zanizalo ogon.

Siatka zalatana pochodzi z patch_scene_ceiling.py; kontrola domkniecia (ray_escape_survey.py
z --mesh) pokazala spadek ucieczki promieni z 21.79 % do 0.00 % mediany.

Uruchomienie (2 konstrukcje Simulatora w jednym procesie, limit ~30 na proces):
    python my-operations/measurements/ceiling_patch_rt60.py --scene frl_apartment_2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # dla soundspaces1_rt60
import quaternion  # noqa: F401  (przed habitat_sim — habitat-sim/local_changes.patch)
import habitat_sim

from echo_core import audio
from echo_core.params import INDIRECT_RAY_COUNT, SENSOR_HEIGHT, THREAD_COUNT
from echo_core.paths import MATERIAL_CONFIG, OUT_ROOT, REPO_ROOT, scene_mesh
from echo_core.scenes import load_scene_locations
from soundspaces1_rt60 import BANDS, octave_sos
from scipy.signal import sosfilt

PATCHED_ROOT = REPO_ROOT / "outputs/patched_scenes"
OUT_DIR = REPO_ROOT / "outputs/measurements/ceiling_patch"
SAMPLE_RATE = 44100

# Mediany RT60 z prekomputowanych RIR-ow SoundSpaces 1.0, ten sam filtr par
# (odleglosc 1-3 m) i ten sam estymator — policzone przez soundspaces1_rt60.py,
# surowy wydruk: outputs/measurements/ss1_verdict.txt.
SS1_RT60 = {
    "frl_apartment_2": {250: 0.493, 500: 0.520, 1000: 0.463, 2000: 0.429},
    "office_1":        {250: 0.418, 500: 0.417, 1000: 0.396, 2000: 0.390},
}


def t60_from_energy(e, fs, lo_db=-5.0, hi_db=-25.0):
    """Schroeder -> T20 -> RT60 na juz przefiltrowanej i usrednionej ENERGII.

    Identyczne okno i obcięcie co soundspaces1_rt60.t60() zastosowane do danych
    SS 1.0 — rozni sie tylko tym, ze przyjmuje energie usredniona po renderach
    zamiast pojedynczego przebiegu cisnienia.
    """
    edc = np.cumsum(e[::-1])[::-1]
    if edc[0] <= 0:
        return np.nan
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-300))
    db = db[:int(0.90 * len(db))]     # RIR z RLR jest UCIETY, nie gasnie w szum
    if db.min() > hi_db:
        return np.nan
    i0, i1 = int(np.argmax(db <= lo_db)), int(np.argmax(db <= hi_db))
    if i1 <= i0 + 10:
        return np.nan
    slope = np.polyfit(np.arange(i0, i1) / fs, db[i0:i1], 1)[0]
    return float(-60.0 / slope) if slope < 0 else np.nan


def build(mesh_path):
    class _A:
        pass
    a = _A()
    a.scene = str(mesh_path)
    a.sensor_height = SENSOR_HEIGHT
    a.material_config = str(MATERIAL_CONFIG)
    a.out_dir = str(OUT_ROOT / "_rlr_scratch")
    a.indirect_ray_count = INDIRECT_RAY_COUNT
    a.thread_count = THREAD_COUNT
    a.gpu_device_id = 0
    return audio.build_simulator(a)


def render_rir(sim, rx, tx, material_pending):
    """Jeden render z ZRODLEM ODDALONYM. -> (rir mono float64, czy config juz podany)

    `get_sensor_observations()` ustawia wylacznie transform SLUCHACZA (patrz
    echo_core.audio.phase3_echolocation), wiec zrodlo ustawione tutaj zostaje
    tam, gdzie je postawiono.
    """
    st = habitat_sim.AgentState()
    st.position = rx
    st.rotation = habitat_sim.utils.common.quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)

    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if material_pending:
        sensor.setAudioMaterialsJSON(str(MATERIAL_CONFIG))
    listener = np.array(sensor.node.absolute_translation)
    sensor.setAudioSourceTransform(np.array(tx, dtype=np.float32))
    sensor.setAudioListenerTransform(listener, quaternion.as_float_array(st.rotation))

    obs = sim.get_sensor_observations()
    rir = np.transpose(np.array(obs["audio_sensor"]))
    if rir.size == 0 or not np.any(rir):
        raise RuntimeError(f"pusty RIR dla odbiornika {rx} / zrodla {tx}")
    return rir.mean(axis=1).astype(np.float64), False


def measure(sim, pairs, positions, n_renders, warmup, label):
    """-> {pasmo: [RT60 per para]}, dlugosc RIR"""
    mp = True
    rx0 = positions[pairs[0][0]] .copy()
    t0 = time.perf_counter()
    for _ in range(warmup):
        _, mp = render_rir(sim, rx0, rx0 + np.array([1.0, 0, 0], np.float32), mp)
    print(f"    [{label}] rozgrzewka {warmup} renderow w {time.perf_counter()-t0:.1f} s")

    sos = {fc: octave_sos(fc, SAMPLE_RATE) for fc in BANDS}
    out = {fc: [] for fc in BANDS}
    n_rir = 0
    t0 = time.perf_counter()
    for k, (a, b) in enumerate(pairs):
        rx = positions[a] + np.array([0.0, 0.0, 0.0], np.float32)
        tx = positions[b] + np.array([0.0, SENSOR_HEIGHT, 0.0], np.float32)
        per_render = []
        for _ in range(n_renders):
            h, mp = render_rir(sim, rx, tx, mp)
            n_rir = len(h)
            # filtr w domenie CISNIENIA osobno dla kazdego renderu, potem kwadrat,
            # dopiero potem usrednienie — inaczej gina fazy odbic stochastycznych
            per_render.append({fc: sosfilt(sos[fc], h) ** 2 for fc in BANDS})
        # RLR zwraca RIR-y o ROZNEJ dlugosci miedzy renderami (ucina adaptacyjnie),
        # wiec obcinamy do wspolnej. Dopelnianie zerami zanizaloby ogon: render,
        # ktory skonczyl sie wczesniej, wnosilby do sredniej zero energii tam,
        # gdzie pozostale jeszcze zanikaja.
        n_common = min(len(r[BANDS[0]]) for r in per_render)
        acc = {fc: np.sum([r[fc][:n_common] for r in per_render], axis=0) for fc in BANDS}
        for fc in BANDS:
            v = t60_from_energy(acc[fc] / n_renders, SAMPLE_RATE)
            if np.isfinite(v):
                out[fc].append(v)
        if (k + 1) % 20 == 0:
            print(f"    [{label}] {k+1}/{len(pairs)} par, {time.perf_counter()-t0:.0f} s")
    return out, n_rir


def sample_pairs(positions, loc_ids, n_pairs, dmin, dmax, seed):
    ids = np.array(loc_ids)
    P = np.stack([positions[i] for i in ids])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    ii, jj = np.where((d >= dmin) & (d <= dmax))
    cand = [(int(ids[i]), int(ids[j])) for i, j in zip(ii, jj) if i < j]
    if not cand:
        raise RuntimeError(f"brak par w przedziale {dmin}-{dmax} m")
    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(cand))[:n_pairs]
    pairs = [cand[i] for i in sel]
    dd = [float(np.linalg.norm(positions[a] - positions[b])) for a, b in pairs]
    return pairs, (float(np.mean(dd)), float(np.min(dd)), float(np.max(dd)), len(cand))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="frl_apartment_2")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--renders", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--dmax", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--control", action="store_true",
                    help="tylko oryginalna siatka (scena naturalnie zamknieta)")
    args = ap.parse_args()

    patched = PATCHED_ROOT / args.scene / "habitat/mesh_semantic.ply"
    if not args.control and not patched.exists():
        sys.exit(f"brak zalatanej sceny: {patched}\nnajpierw: patch_scene_ceiling.py --scene {args.scene}")
    if args.scene not in SS1_RT60:
        sys.exit(f"brak wartosci SS 1.0 dla {args.scene} — pobierz RIR-y i przelicz "
                 f"soundspaces1_rt60.py")

    loc_ids, positions = load_scene_locations(args.scene)
    pairs, dst = sample_pairs(positions, loc_ids, args.pairs, args.dmin, args.dmax, args.seed)
    print(f"\n  {args.scene}: {len(loc_ids)} lokalizacji, par w {args.dmin}-{args.dmax} m: {dst[3]}, "
          f"uzytych: {len(pairs)}")
    print(f"  odleglosc zrodlo-odbiornik: srednia {dst[0]:.2f} m (zakres {dst[1]:.2f}-{dst[2]:.2f})")
    print(f"  renderow na pare: {args.renders}, rozgrzewka: {args.warmup}, "
          f"promieni: {INDIRECT_RAY_COUNT}\n")

    # --control: scena naturalnie zamknieta, bez latania. Sluzy do ustalenia, jak nasz
    # silnik wypada wobec SS 1.0 przy TYM SAMYM estymatorze i tej samej geometrii par —
    # bez tego nie da sie odroznic "lata przestrzelila" od "nasz silnik gra dluzej".
    meshes = [("oryginalna", scene_mesh(args.scene))]
    if not args.control:
        meshes.append(("ZALATANA", patched))

    results = {}
    for label, mesh in meshes:
        print(f"  --- siatka {label}: {mesh}")
        sim = build(mesh)
        try:
            res, n_rir = measure(sim, pairs, positions, args.renders, args.warmup, label)
        finally:
            sim.close()
        results[label] = res
        print(f"    [{label}] RIR {n_rir} probek = {n_rir/SAMPLE_RATE:.2f} s\n")

    ss1 = SS1_RT60[args.scene]
    has_patch = "ZALATANA" in results

    print("=" * 78)
    print(f"  WYNIK — RT60 [s], {args.scene}, zrodlo oddalone {args.dmin}-{args.dmax} m")
    print("=" * 78)
    if has_patch:
        print(f"  {'pasmo':>7}{'oryginalna':>13}{'ZALATANA':>11}{'zmiana':>9}"
              f"{'SS 1.0':>10}{'orig/SS1':>10}{'zalat/SS1':>11}")
    else:
        print(f"  {'pasmo':>7}{'nasze':>13}{'SS 1.0':>10}{'nasze/SS1':>12}")
    rows = []
    for fc in BANDS:
        o = np.median(results["oryginalna"][fc]) if results["oryginalna"][fc] else np.nan
        s = ss1[int(fc)]
        row = {"band_hz": int(fc), "orig": round(float(o), 4), "ss1": s,
               "ratio_orig_ss1": round(float(o / s), 4)}
        if has_patch:
            p = np.median(results["ZALATANA"][fc]) if results["ZALATANA"][fc] else np.nan
            print(f"  {int(fc):>7}{o:>13.3f}{p:>11.3f}{p/o:>8.2f}x{s:>10.3f}"
                  f"{o/s:>10.2f}x{p/s:>11.2f}x")
            row.update(patched=round(float(p), 4),
                       ratio_patched_orig=round(float(p / o), 4),
                       ratio_patched_ss1=round(float(p / s), 4))
        else:
            print(f"  {int(fc):>7}{o:>13.3f}{s:>10.3f}{o/s:>11.2f}x")
        rows.append(row)

    mid = [r for r in rows if r["band_hz"] in (500, 1000, 2000)]
    ro = float(np.median([r["ratio_orig_ss1"] for r in mid]))
    print("\n  Mediana stosunku do SS 1.0 (500 Hz - 2 kHz):")
    if has_patch:
        rp = float(np.median([r["ratio_patched_ss1"] for r in mid]))
        print(f"    oryginalna: {ro:.2f}x        zalatana: {rp:.2f}x")
        # porownanie w skali logarytmicznej: 0.5x i 2.0x to ten sam blad co do wielkosci
        better = abs(np.log(rp)) < abs(np.log(ro))
        print(f"\n  ==> Domkniecie sufitem {'ZBLIZA' if better else 'NIE zbliza'} nasze RT60 do "
              f"SoundSpaces 1.0.")
    else:
        rp = None
        print(f"    {ro:.2f}x   (KONTROLA: scena naturalnie zamknieta, bez latania)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.scene}_rt60{'_control' if args.control else ''}.json"
    ratios = {"orig": round(ro, 4)}
    if rp is not None:
        ratios["patched"] = round(rp, 4)
    out.write_text(json.dumps({
        "scene": args.scene, "control_only": args.control, "pairs": len(pairs),
        "renders_per_pair": args.renders, "warmup": args.warmup,
        "dist_mean_m": round(dst[0], 3),
        "dist_range_m": [round(dst[1], 3), round(dst[2], 3)],
        "indirect_ray_count": INDIRECT_RAY_COUNT, "bands": rows,
        "median_ratio_to_ss1_500_2k": ratios,
    }, indent=2))
    print(f"\n  zapisano: {out}")


if __name__ == "__main__":
    main()
