#!/usr/bin/env python3
"""TEST FIZYCZNY: czas pogłosu z symulacji vs wzory Sabine'a i Eyringa.

Jedyny test w projekcie, ktory sprawdza FIZYKE, a nie spojnosc wewnetrzna.
`--verify` kontroluje ksztalty, dtype, NaN i regule adaptacyjna — czyli czy
generator zrobil to, co mial. Ten skrypt pyta, czy wynik ma sens akustyczny.

Obie strony rownania pochodza z NIEZALEZNYCH zrodel:

  strona A (geometria + materialy, bez symulacji akustycznej)
      powierzchnie per kategoria  <- mesh_semantic.ply (parser z tools/replica_semantic_area.py)
      kategoria -> material       <- replica_material_config.json (pole `labels`)
      absorpcja alfa(f)           <- ten sam config, pasma oktawowe 125-4000 Hz
      objetosc V                  <- rzut z kategorii `ceiling` x (srednie z sufitu - srednie z podlogi)
                                     UWAGA: surowy PLY jest Z-UP, habitat transformuje do y-up
                                     dopiero przy ladowaniu — sprawdzone na zakresach wspolrzednych
      Sabine:  RT60 = 0.161 * V / A,          A = suma S_i * alfa_i(f)
      Eyring:  RT60 = 0.161 * V / (-S * ln(1 - alfa_sr))    [wlasciwszy przy alfa_sr > 0.2]

  strona B (symulacja akustyczna)
      M renderow -> usrednienie ENERGII h^2(t) (nie amplitudy — szum MC jest
      nieskorelowany w fazie, wiec usrednianie amplitudy tlumiloby ogon)
      filtr oktawowy -> calkowanie wsteczne Schroedera -> EDC w dB
      T20/T30 z dopasowania prostej, ekstrapolowane do 60 dB

CZEGO TEN TEST NIE DOWODZI. Sabine zaklada pole dyfuzyjne i w miare rownomierna
absorpcje w pomieszczeniu zblizonym do prostopadloscianu. Sceny Replica to
umeblowane, otwarte przestrzenie o skrajnie nierownomiernej absorpcji (Curtain
ma alfa = 0.75 przy 1 kHz, Tile 0.01). Zgodnosc co do czynnika ~2 jest tu
normalna i oczekiwana. Test wykrywa bledy RZEDU WIELKOSCI — materialy w ogole
nieprzypisane, zla jednostka, obciety RIR — a nie kalibruje dokladnosci.

KONTROLA NEGATYWNA (darmowa): ta sama Sabine policzona przy zalozeniu, ze KAZDA
powierzchnia ma material domyslny (alfa = 0.1 plasko). Jesli materialy naprawde
wplywaja na symulacje, pomiar powinien lezec blizej wariantu z prawdziwymi
materialami niz z domyslnym.

Raport: RAPORT_SESJI_2026-07-26_29.md §5 pkt 1 (luka, ktora ten skrypt zamyka)

Uruchomienie:
    conda activate habitat
    python my-operations/measurements/rt60_vs_sabine.py
    python my-operations/measurements/rt60_vs_sabine.py --scenes room_0 office_1
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim  # noqa: F401

from echo_core import audio, spectrogram
from echo_core.params import INDIRECT_RAY_COUNT, SENSOR_HEIGHT, THREAD_COUNT, WARMUP_DISCARD
from echo_core.paths import (MATERIAL_CONFIG, OUT_ROOT, REPO_ROOT, SCENE_ROOT,
                             scene_mesh)
from echo_core.scenes import load_scene_locations

BANDS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
DEFAULT_SCENES = ("room_0", "office_1", "hotel_0", "frl_apartment_0", "apartment_0")
M_RENDERS = 30
SABINE_K = 0.161            # [s*m^-1], stala Sabine'a dla powietrza w 20 st. C

FACE_DTYPE = np.dtype([("n", "u1"), ("v", "<u4", 4), ("object_id", "<u2")])
VERT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])


# --------------------------------------------------------------------------
# STRONA A: geometria i materialy
# --------------------------------------------------------------------------
def _header_counts(buf):
    head = buf[: buf.find(b"end_header\n") + len(b"end_header\n")].decode("ascii", "replace")
    counts = {}
    for line in head.splitlines():
        if line.startswith("element "):
            _, name, num = line.split()
            counts[name] = int(num)
    return len(head), counts


def mesh_path(scene, patched=False):
    """-> sciezka do siatki: oryginalnej albo ZALATANEJ (patch_scene_holes.py).

    Wariant zalatany MUSI byc uzyty po obu stronach rownania: i do geometrii
    (dolozone ~90 m2 sufitu wchodzi do S w Sabine), i do renderowania. Podanie go
    tylko z jednej strony dawaloby wynik bez sensu.
    """
    if patched:
        p = REPO_ROOT / "outputs/patched_scenes" / scene / "habitat/mesh_semantic.ply"
        if not p.exists():
            raise SystemExit(f"brak zalatanej sceny {p}\n"
                             f"najpierw: patch_scene_holes.py --scene {scene}")
        return p
    return SCENE_ROOT / scene / "habitat/mesh_semantic.ply"


def scene_geometry(scene, patched=False):
    """-> (pole per kategoria [m2], rzut [m2], wysokosc [m], pokrycie sufitem [0-1])

    Parser identyczny z tools/replica_semantic_area.py: Replica zapisuje same
    quady o stalej dlugosci rekordu, wiec da sie je wczytac jednym frombuffer.
    """
    sd = SCENE_ROOT / scene
    buf = mesh_path(scene, patched).read_bytes()
    off, counts = _header_counts(buf)
    nv, nf = counts["vertex"], counts["face"]
    verts = np.frombuffer(buf, dtype=VERT_DTYPE, count=nv, offset=off)
    faces = np.frombuffer(buf, dtype=FACE_DTYPE, count=nf,
                          offset=off + nv * VERT_DTYPE.itemsize)
    if not np.all(faces["n"] == 4):
        raise RuntimeError(f"{scene}: nie same quady, parser wymaga rozszerzenia")

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    v = faces["v"]
    p0, p1, p2, p3 = xyz[v[:, 0]], xyz[v[:, 1]], xyz[v[:, 2]], xyz[v[:, 3]]
    area = 0.5 * (np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
                  + np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1))
    # UWAGA na konwencje osi: surowy mesh_semantic.ply jest Z-UP (rozpietosc osi z
    # to 2.8-3.1 m we wszystkich scenach, czyli wysokosc pomieszczenia). Habitat
    # stosuje transformacje do y-up dopiero przy ladowaniu sceny. Sprawdzone
    # bezposrednio na zakresach wspolrzednych — patrz tez PKL_FORMAT.md, gdzie
    # konwersja points.txt brzmi "habitat x = a, habitat z = -b".
    zmean = 0.25 * (p0[:, 2] + p1[:, 2] + p2[:, 2] + p3[:, 2])

    info = json.loads((sd / "habitat/info_semantic.json").read_text())
    cat_of = {o["id"]: ("<class_id=-1>" if o["class_id"] == -1 else o["class_name"])
              for o in info["objects"]}
    oid = faces["object_id"].astype(np.int64)

    by_cat = collections.Counter()
    z_by_cat = collections.defaultdict(list)
    cats = np.array([cat_of.get(int(o), "<brak w info_semantic>") for o in oid])
    for c in np.unique(cats):
        sel = cats == c
        by_cat[str(c)] += float(area[sel].sum())
        z_by_cat[str(c)] = (zmean[sel], area[sel])

    # Rzut poziomy pomieszczenia: SUFIT, a nie podloga. Kategoria `floor` jest
    # przyslonieta przez dywany, maty i meble stojace bezposrednio na podlodze
    # (osobne kategorie `rug`, `mat`), wiec zaniza rzut. Sufity w skanach Replica
    # sa praktycznie nieprzyslonieta plaszczyzna.
    ceil_area = by_cat.get("ceiling", 0.0)
    floor_like = sum(by_cat.get(c, 0.0) for c in ("floor", "rug", "mat", "stair"))
    footprint = max(ceil_area, floor_like)

    def _wmean_z(cat):
        if cat not in z_by_cat or len(z_by_cat[cat][0]) == 0:
            return None
        z, w = z_by_cat[cat]
        return float(np.average(z, weights=w))

    z_floor, z_ceil = _wmean_z("floor"), _wmean_z("ceiling")
    if z_floor is None or z_ceil is None:
        height = float(xyz[:, 2].max() - xyz[:, 2].min())   # awaryjnie: rozpietosc osi z
    else:
        height = abs(z_ceil - z_floor)
    ceil_ratio = ceil_area / footprint if footprint else 0.0
    return dict(by_cat), footprint, height, ceil_ratio


def absorption_table():
    """kategoria -> (nazwa materialu, alfa w pasmach BANDS). Brak dopasowania -> Default."""
    cfg = json.loads(MATERIAL_CONFIG.read_text())
    default = None
    table = {}
    for m in cfg["materials"]:
        a = m["absorption"]
        freqs, vals = np.array(a[0::2], dtype=float), np.array(a[1::2], dtype=float)
        alpha = np.interp(BANDS, freqs, vals)          # poza zakresem -> wartosc skrajna
        if m["name"] == "Default":
            default = (m["name"], alpha)
        for lab in m.get("labels", []):
            table[lab] = (m["name"], alpha)
    if default is None:
        raise RuntimeError("brak materialu 'Default' w configu")
    return table, default


def sabine_eyring(by_cat, volume, table, default, force_default=False):
    """-> (RT60 Sabine [s] per pasmo, RT60 Eyring, laczna powierzchnia S, alfa srednie)"""
    S = sum(by_cat.values())
    A = np.zeros(len(BANDS))
    for cat, area in by_cat.items():
        _name, alpha = default if force_default else table.get(cat, default)
        A += area * alpha
    alpha_bar = A / S
    rt_sabine = SABINE_K * volume / A
    # Eyring: wlasciwszy przy duzej absorpcji, bo Sabine nie zbiega do zera przy alfa -> 1
    rt_eyring = SABINE_K * volume / (-S * np.log(np.maximum(1.0 - alpha_bar, 1e-6)))
    return rt_sabine, rt_eyring, S, alpha_bar


# --------------------------------------------------------------------------
# STRONA B: pomiar z RIR
# --------------------------------------------------------------------------
def octave_sos(fc, fs):
    lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
    hi = min(hi, 0.99 * fs / 2)
    return butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")


def t60_from_band_energy(e, fs, lo_db=-25.0, hi_db=-45.0):
    """Calkowanie wsteczne Schroedera na GOTOWEJ energii pasmowej.

    -> (rt60, uzyteczny zakres dB, ok)

    UWAGA 1: energia MUSI byc juz przefiltrowana pasmowo w domenie CISNIENIA, tzn.
    filtr nalozony na h(t), a dopiero potem kwadrat. Filtrowanie sqrt(energii)
    (czyli |h| po usrednieniu) jest bledne — usuwa faze, wiec widmo takiego
    sygnalu nie ma nic wspolnego z widmem odpowiedzi impulsowej.

    UWAGA 2 — dlaczego okno dopasowania to -25..-45 dB, a nie klasyczne -5..-25.
    W echolokacji ZRODLO JEST WSPOLLOKOWANE Z ODBIORNIKIEM, wiec dzwiek
    bezposredni ma energie nieporownywalnie wieksza od pola pogłosowego. Krzywa
    Schroedera ma przez to DWA nachylenia: strome na starcie (opadanie samego
    impulsu bezposredniego i wczesnych odbic) i lagodniejsze pozniej (wlasciwy
    pogłos). Zmierzone na apartment_0 przy 1 kHz:
        -5..-15 dB  ->  -136 dB/s  ->  RT60 = 0.441 s   <- zdominowane przez impuls
       -15..-25 dB  ->   -82 dB/s  ->  RT60 = 0.732 s
       -25..-35 dB  ->   -80 dB/s  ->  RT60 = 0.755 s   <- wlasciwy pogłos
    Punkt -5 dB wypada juz po 3 ms, czyli zanim pole pogłosowe zdazy sie rozwinac.
    Klasyczne T20 mierzyloby tu zanik impulsu, nie pomieszczenia — stad okno
    przesuniete w pozna czesc krzywej.
    """
    edc = np.cumsum(e[::-1])[::-1]                   # calka od t do konca
    if edc[0] <= 0:
        return np.nan, 0.0, False
    edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-300))

    # RIR z RLR jest URWANY, a nie zanika w szum: ostatnie probki sa dokladnie
    # zerowe, wiec EDC spada tam do -inf. Taki "zakres dynamiki" jest artefaktem
    # obciecia, nie wlasnoscia pola akustycznego. Uzyteczny zakres liczymy do
    # momentu, w ktorym EDC przestaje byc gladkie — bierzemy 90 % dlugosci IR.
    usable = int(0.90 * len(edc_db))
    edc_db = edc_db[:usable]
    dyn = float(edc_db.min())
    if dyn > hi_db:                                  # za maly zakres na ten fit
        hi_db = max(dyn + 3.0, lo_db - 5.0)
        if hi_db >= lo_db - 4.0:
            return np.nan, dyn, False
    i0 = int(np.argmax(edc_db <= lo_db))
    i1 = int(np.argmax(edc_db <= hi_db))
    if i1 <= i0 + 10:
        return np.nan, dyn, False
    t = np.arange(i0, i1) / fs
    slope, _icpt = np.polyfit(t, edc_db[i0:i1], 1)   # dB/s, ujemne
    if slope >= 0:
        return np.nan, dyn, False
    return float(-60.0 / slope), dyn, True


class _Args:
    def __init__(self, scene, patched=False):
        self.scene = str(mesh_path(scene, patched))
        self.sensor_height = SENSOR_HEIGHT
        self.material_config = str(MATERIAL_CONFIG)
        self.out_dir = str(OUT_ROOT / "_measurement_scratch")
        self.indirect_ray_count = INDIRECT_RAY_COUNT
        self.thread_count = THREAD_COUNT
        self.gpu_device_id = 0


def measure_band_energy(scene, m, warmup, patched=False):
    """-> (energia per pasmo {fc: h_f^2(t) usrednione}, fs, liczba renderow, dlugosc)

    Kolejnosc operacji jest istotna: dla KAZDEGO renderu z osobna filtrujemy
    cisnienie h(t) w pasmie oktawowym, dopiero potem podnosimy do kwadratu
    i usredniamy po renderach. Usrednianie energii (a nie amplitudy) jest
    konieczne, bo szum Monte Carlo jest nieskorelowany w fazie — usrednianie
    amplitudy tlumiloby ogon odpowiedzi.
    """
    ids, positions = load_scene_locations(scene)
    pos = positions[ids[len(ids) // 2]]              # pozycja srodkowa listy
    mc = str(MATERIAL_CONFIG)
    fs = spectrogram.SAMPLE_RATE
    sos = {fc: octave_sos(fc, fs) for fc in BANDS}
    sim = audio.build_simulator(_Args(scene, patched))
    try:
        for _ in range(warmup):
            audio.phase3_echolocation(sim, pos, 0.0, mc, run_simulation=False)
        acc, n, L = {fc: None for fc in BANDS}, 0, None
        for _ in range(m):
            obs, _l, _r = audio.phase3_echolocation(sim, pos, 0.0, mc, run_simulation=False)
            rir = np.array(obs["audio_sensor"], dtype=np.float64)      # (kanaly, probki)
            h = rir.mean(axis=0)                     # mono: srednie cisnienie obu uszu
            L = len(h) if L is None else min(L, len(h))
            for fc in BANDS:
                e = sosfilt(sos[fc], h) ** 2
                acc[fc] = e if acc[fc] is None else (acc[fc][:min(len(acc[fc]), len(e))]
                                                     + e[:min(len(acc[fc]), len(e))])
            n += 1
        return {fc: acc[fc][:L] / n for fc in BANDS}, fs, n, L
    finally:
        sim.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    ap.add_argument("--renders", type=int, default=M_RENDERS)
    # Rozgrzewka moze byc mala: wplywa na SZUM pojedynczego renderu, a tu
    # usredniamy energie z M renderow i mierzymy NACHYLENIE zaniku, nie wariancje.
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--patched", action="store_true",
                    help="uzyj siatek zalatanych z outputs/patched_scenes/")
    args = ap.parse_args()

    table, default = absorption_table()
    print(f"  pasma oktawowe [Hz]: {[int(b) for b in BANDS]}")
    print(f"  renderow na scene: {args.renders}, rozgrzewka: {args.warmup}\n")

    summary = []
    for si, scene in enumerate(args.scenes, 1):
        by_cat, floor_area, height, ceil_ratio = scene_geometry(scene, args.patched)
        closed = ceil_ratio >= 0.85
        volume = floor_area * height
        rt_sab, rt_eyr, S, alpha_bar = sabine_eyring(by_cat, volume, table, default)
        rt_def, _, _, alpha_def = sabine_eyring(by_cat, volume, table, default,
                                                force_default=True)

        bands, fs, n, L = measure_band_energy(scene, args.renders, args.warmup,
                                              args.patched)
        meas, early, dyns = [], [], []
        for fc in BANDS:
            rt, dyn, ok = t60_from_band_energy(bands[fc], fs)            # pozne: -25..-45
            rt_e, _, ok_e = t60_from_band_energy(bands[fc], fs, -5.0, -25.0)   # klasyczne T20
            meas.append(rt if ok else np.nan)
            early.append(rt_e if ok_e else np.nan)
            dyns.append(dyn)
        meas, early = np.array(meas), np.array(early)

        print(f"[{si}/{len(args.scenes)}] {scene}")
        print(f"  geometria: rzut {floor_area:7.1f} m2, wysokosc {height:5.2f} m, "
              f"V = {volume:8.1f} m3, S = {S:8.1f} m2")
        print(f"  pokrycie sufitem: {100*ceil_ratio:5.1f} %  -> "
              + ("scena ZAMKNIETA, Sabine/Eyring stosowalne"
                 if closed else
                 "scena OTWARTA (brak sufitu w skanie) — Sabine/Eyring NIE stosuja sie,\n"
                 "                          bo zakladaja zamknieta objetosc; energia ucieka gora"))
        print(f"  RIR: {L} probek = {L/fs:.2f} s, usrednione z {n} renderow")
        print(f"  {'pasmo':>8}{'alfa sr.':>10}{'Sabine':>9}{'Eyring':>9}{'T20 wcz.':>10}"
              f"{'POZNE':>9}{'pozne/Eyring':>14}")
        for i, fc in enumerate(BANDS):
            ratio = meas[i] / rt_eyr[i] if np.isfinite(meas[i]) else np.nan
            m_s = f"{meas[i]:.3f}" if np.isfinite(meas[i]) else "  n/d"
            e_s = f"{early[i]:.3f}" if np.isfinite(early[i]) else "  n/d"
            r_s = f"{ratio:.2f}x" if np.isfinite(ratio) else "  n/d"
            print(f"  {int(fc):>8}{alpha_bar[i]:>10.3f}{rt_sab[i]:>9.3f}{rt_eyr[i]:>9.3f}"
                  f"{e_s:>10}{m_s:>9}{r_s:>14}")
        print(f"  kontrola negatywna (wszystko material domyslny, alfa={alpha_def[3]:.2f}): "
              f"Sabine przy 1 kHz = {rt_def[3]:.3f} s")
        summary.append((scene, meas, rt_sab, rt_eyr, rt_def, volume, closed))
        print()

    # --- zbiorczo -----------------------------------------------------------
    print("=" * 96)
    print("  PODSUMOWANIE — stosunek zmierzone / Eyring (pasma 500 Hz - 2 kHz)")
    print("=" * 96)
    print(f"  {'scena':<20}{'sufit':>7}{'V [m3]':>9}{'500 Hz':>10}{'1 kHz':>10}{'2 kHz':>10}"
          f"{'1 kHz: pomiar':>15}{'Eyring':>9}{'domyslny':>10}")
    ratios = []
    for scene, meas, rt_sab, rt_eyr, rt_def, volume, closed in summary:
        r = [meas[i] / rt_eyr[i] for i in (2, 3, 4)]
        if closed:
            ratios += [x for x in r if np.isfinite(x)]
        cells = "".join(f"{x:>10.2f}" if np.isfinite(x) else f"{'n/d':>10}" for x in r)
        m1 = f"{meas[3]:.3f}" if np.isfinite(meas[3]) else "n/d"
        tag = "zamk." if closed else "BRAK"
        print(f"  {scene:<20}{tag:>7}{volume:>9.0f}{cells}{m1:>15}{rt_eyr[3]:>9.3f}{rt_def[3]:>10.3f}")
    if ratios:
        ratios = np.array(ratios)
        print(f"\n  stosunek zmierzone/Eyring, TYLKO SCENY ZAMKNIETE: "
              f"mediana {np.median(ratios):.2f}x, zakres {ratios.min():.2f}-{ratios.max():.2f}x "
              f"(n={len(ratios)})")
        print(f"\n  Sceny bez sufitu wylaczone z agregatu — Sabine/Eyring zakladaja zamknieta")
        print(f"  objetosc, wiec dla nich nie sa wlasciwym odniesieniem, a nie 'nie zgadzaja sie'.")
        print(f"  Zgodnosc co do czynnika ~2 jest dla scen umeblowanych NORMALNA: oba wzory")
        print(f"  zakladaja pole dyfuzyjne i rownomierna absorpcje. Test wykrywa bledy rzedu")
        print(f"  wielkosci (materialy nieprzypisane, zla jednostka, urwany RIR), nie kalibruje.")


if __name__ == "__main__":
    main()
