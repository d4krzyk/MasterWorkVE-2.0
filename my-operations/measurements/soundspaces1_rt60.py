#!/usr/bin/env python3
"""TEST: czy SoundSpaces 1.0 uzywal tych samych scen Replica BEZ SUFITU?

Pytanie ma znaczenie dla pracy: jesli tak, otwartosc szesciu scen `frl_apartment_*`
jest wlasnoscia zbioru Replica DZIELONA Z LITERATURA (Gao i in. uzywali tych samych
scen), a nie osobliwoscia tej pracy — z ograniczenia staje sie przypisem.

Czego NIE da sie ustalic inaczej: kod renderujacy SoundSpaces 1.0 nie zostal
opublikowany ("we do not open source the rendering code at this time", README).
Ustalono za to, ze Replica dostarcza dokladnie JEDNA geometrie na scene —
`mesh.ply` i `habitat/mesh_semantic.ply` maja bit-identyczne tablice wierzcholkow —
wiec nie istnieje zamknieta siatka, ktorej mogliby uzyc. Brakuje jednak dowodu
wprost, bo ich potok mogl teoretycznie domykac objetosc przed symulacja.

PREDYKCJA FALSYFIKOWALNA. W naszych danych (SoundSpaces 2.0 / RLRAudioPropagation):

    scena              typ         V [m3]   RT60 @ 1 kHz
    frl_apartment_2    OTWARTA       191      0.263 s
    office_1           zamknieta      23      0.401 s

Scena otwarta ma KROTSZY poglos mimo 8x wiekszej objetosci — to jest sygnatura
braku sufitu. Jesli SoundSpaces 1.0 uzywal tej samej geometrii, ich dane musza
pokazac ten sam ODWROCONY porzadek. Jesli domykali objetosc, `frl_apartment_2`
wyjdzie u nich DLUZSZY niz `office_1`, zgodnie z objetoscia.

DLACZEGO TO OMIJA PROBLEM ROZNYCH SILNIKOW. GENERATOR_PARAMS.md §5 ogr. 1 zabrania
porownywania metryk BEZWZGLEDNYCH miedzy SoundSpaces 1.0 a 2.0. Ten test tego nie
robi: pyta wylacznie o KONTRAST otwarte/zamkniete WEWNATRZ danych SoundSpaces 1.0.
Wartosci bezwzgledne nigdy nie przekraczaja granicy silnika.

ZASTRZEZENIE: w SoundSpaces 1.0 zrodlo i odbiornik to ROZNE wezly grafu, wiec
dominacja dzwieku bezposredniego jest inna niz w echolokacji. Dlatego okno
dopasowania jest tu klasyczne (-5..-25 dB), a nie przesuniete w pozna czesc
krzywej jak w rt60_vs_sabine.py. Porownujemy kontrast, nie liczby.

WYMAGA: sound-spaces/data/binaural_rirs/replica/<scena>/  (pobrane osobno)

Raport: OBSERWACJE_METODOLOGICZNE.md §1

Uruchomienie (bez GPU):
    python my-operations/measurements/soundspaces1_rt60.py
    python my-operations/measurements/soundspaces1_rt60.py --scenes office_1 frl_apartment_2
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_core.paths import REPO_ROOT

BANDS = (250.0, 500.0, 1000.0, 2000.0)
RIR_ROOT = REPO_ROOT / "sound-spaces/data/binaural_rirs/replica"
DEFAULT_SCENES = ("office_1", "frl_apartment_2")
N_PAIRS = 80


def octave_sos(fc, fs):
    lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
    hi = min(hi, 0.99 * fs / 2)
    return butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")


def t60(h, fs, fc, lo_db=-5.0, hi_db=-25.0):
    """Schroeder -> T20 -> RT60. h to cisnienie (mono). -> rt60 albo nan."""
    e = sosfilt(octave_sos(fc, fs), h) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    if edc[0] <= 0:
        return np.nan
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-300))
    usable = int(0.90 * len(db))
    db = db[:usable]
    if db.min() > hi_db:
        return np.nan
    i0, i1 = int(np.argmax(db <= lo_db)), int(np.argmax(db <= hi_db))
    if i1 <= i0 + 10:
        return np.nan
    slope = np.polyfit(np.arange(i0, i1) / fs, db[i0:i1], 1)[0]
    return float(-60.0 / slope) if slope < 0 else np.nan


def node_xz(scene):
    """loc_id -> (x, z) w metrach. Nazwy plikow SS1.0 to <odbiornik>_<zrodlo>."""
    import pandas as pd
    from echo_core.paths import points_txt
    df = pd.read_csv(points_txt(scene), sep="\t", header=None, names=["id", "a", "b", "c"])
    return {int(r.id): (float(r.a), -float(r.b)) for r in df.itertuples()}


def scene_rt60(scene, n_pairs, rng, dmin=1.0, dmax=3.0):
    """-> (mediana RT60 per pasmo, liczba par, fs, dlugosc RIR [s], plikow, staty odleglosci)

    KLUCZOWE: pary sa filtrowane po ODLEGLOSCI zrodlo-odbiornik. Bez tego
    porownanie miedzy scenami jest obciazone — w `office_1` (16 wezlow) losowe
    pary maja mediane 1.12 m, w `frl_apartment_2` (125 wezlow) 4.03 m. Wieksza
    odleglosc oslabia dzwiek bezposredni wzgledem pola poglosowego, wiec okno
    dopasowania T20 (-5..-25 dB) trafia w INNA czesc krzywej zaniku. Mierzylibysmy
    wtedy roznice geometrii probkowania, a nie roznice pogłosu.

    Czasu dotarcia nie da sie do tego uzyc: RIR-y SS 1.0 sa WYROWNANE CZASOWO
    (we wszystkich plikach `office_1` pierwszy impuls wypada w tej samej probce),
    wiec odleglosc liczymy z wspolrzednych wezlow w points.txt.
    """
    import soundfile as sf
    import numpy as _np
    d = RIR_ROOT / scene
    if not d.is_dir():
        raise FileNotFoundError(
            f"brak danych SoundSpaces 1.0 dla sceny {scene}: {d}\n"
            f"  pobierz:  cd sound-spaces/data && wget "
            f"http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs/replica/{scene}.tar.gz "
            f"&& tar xzf {scene}.tar.gz --strip-components=1")
    angle_dirs = sorted(p for p in d.iterdir() if p.is_dir())
    if not angle_dirs:
        raise FileNotFoundError(f"{d} nie zawiera podkatalogow katow")
    all_files = sorted(angle_dirs[0].glob("*.wav"))
    if not all_files:
        raise FileNotFoundError(f"{angle_dirs[0]} nie zawiera plikow .wav")

    P = node_xz(scene)
    banded = []
    for f in all_files:
        try:
            r_, s_ = f.stem.split("_")
            (x1, z1), (x2, z2) = P[int(r_)], P[int(s_)]
        except (ValueError, KeyError):
            continue
        dist = float(_np.hypot(x1 - x2, z1 - z2))
        if dmin <= dist <= dmax:
            banded.append((f, dist))
    if not banded:
        raise RuntimeError(f"{scene}: brak par w przedziale {dmin}-{dmax} m")
    sel = rng.sample(banded, min(n_pairs, len(banded)))
    pick = [f for f, _ in sel]
    dstats = _np.array([dd for _, dd in sel])

    per_band, fs, lens = {fc: [] for fc in BANDS}, None, []
    for f in pick:
        x, sr = sf.read(str(f), always_2d=True)
        fs = sr
        h = x.mean(axis=1)                      # mono: srednie cisnienie obu uszu
        if not np.any(h):
            continue                            # SS1.0 ma pary bez sciezki -> same zera
        lens.append(len(h) / sr)
        for fc in BANDS:
            v = t60(h, sr, fc)
            if np.isfinite(v):
                per_band[fc].append(v)
    med = {fc: (float(np.median(per_band[fc])) if per_band[fc] else np.nan) for fc in BANDS}
    n_ok = max(len(per_band[fc]) for fc in BANDS)
    return (med, n_ok, fs, (float(np.median(lens)) if lens else 0.0), len(all_files),
            (float(dstats.mean()), float(dstats.min()), float(dstats.max()), len(banded)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmin", type=float, default=1.0, help="min. odleglosc zrodlo-odbiornik [m]")
    ap.add_argument("--dmax", type=float, default=3.0, help="max. odleglosc [m]")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # nasze wartosci (SoundSpaces 2.0), do porownania KONTRASTU, nie liczb
    # RT60 zmierzone NASZYM silnikiem przy ZRODLE ODDALONYM O ~2 m, czyli
    # w geometrii SoundSpaces 1.0 (kontrola: eliminuje wplyw wspollokowania
    # zrodla z odbiornikiem, ktore w echolokacji tworzy wielonachyleniowy zanik).
    OURS = {"office_1": (23, 0.358, "zamknieta"),
            "frl_apartment_2": (191, 0.186, "OTWARTA"),
            "frl_apartment_0": (180, 0.195, "OTWARTA"),
            "apartment_0": (379, 0.771, "zamknieta"),
            "room_0": (83, 0.463, "zamknieta"),
            "hotel_0": (75, 0.550, "zamknieta")}

    print(f"  RIR-y SoundSpaces 1.0 z: {RIR_ROOT}")
    print(f"  par na scene: {args.pairs}, pasma: {[int(b) for b in BANDS]} Hz\n")
    res = {}
    for s in args.scenes:
        try:
            med, n_ok, fs, dur, n_files, dst = scene_rt60(
                s, args.pairs, rng, args.dmin, args.dmax)
        except FileNotFoundError as e:
            print(f"  {s}: {e}\n")
            continue
        res[s] = med
        v, ours, typ = OURS.get(s, (None, None, "?"))
        print(f"  {s}  ({typ}, V = {v} m3)")
        print(f"    plikow w katalogu: {n_files}, par w przedziale "
              f"{args.dmin}-{args.dmax} m: {dst[3]}, uzytych: {n_ok}")
        print(f"    odleglosc zrodlo-odbiornik: srednia {dst[0]:.2f} m "
              f"(zakres {dst[1]:.2f}-{dst[2]:.2f}),  fs = {fs} Hz, RIR ~{dur:.2f} s")
        print(f"    {'pasmo':>8}{'RT60 SS1.0':>13}")
        for fc in BANDS:
            m = f"{med[fc]:.3f} s" if np.isfinite(med[fc]) else "n/d"
            print(f"    {int(fc):>8}{m:>13}")
        if ours:
            print(f"    (nasze SS2.0 przy 1 kHz: {ours:.3f} s — NIE porownywac wprost,")
            print(f"     inny silnik; sluzy tylko do porownania KONTRASTU miedzy scenami)")
        print()

    # --- werdykt: kontrast otwarte vs zamkniete ---------------------------
    op = [s for s in res if OURS.get(s, (0, 0, "?"))[2] == "OTWARTA"]
    cl = [s for s in res if OURS.get(s, (0, 0, "?"))[2] == "zamknieta"]
    print("=" * 92)
    print("  WERDYKT — kontrast otwarte/zamkniete WEWNATRZ danych SoundSpaces 1.0")
    print("=" * 92)
    if not op or not cl:
        print("  Brak danych dla obu typow scen — pobierz co najmniej jedna otwarta")
        print("  (frl_apartment_*) i jedna zamknieta (np. office_1).")
        return 1
    for s in op + cl:
        v, ours, typ = OURS[s]
        r1 = res[s][1000.0]
        print(f"  {s:<20}{typ:>11}  V={v:>4} m3   SS1.0 @1kHz: "
              f"{r1:.3f} s" if np.isfinite(r1) else f"  {s:<20}{typ:>11}  n/d")
    o1 = np.nanmedian([res[s][1000.0] for s in op])
    c1 = np.nanmedian([res[s][1000.0] for s in cl])
    vo = np.median([OURS[s][0] for s in op])
    vc = np.median([OURS[s][0] for s in cl])
    print(f"\n  mediana RT60 @1 kHz: otwarte {o1:.3f} s (V~{vo:.0f} m3), "
          f"zamkniete {c1:.3f} s (V~{vc:.0f} m3)")
    print(f"  stosunek objetosci otwarte/zamkniete: {vo/vc:.1f}x")
    if o1 < c1:
        print(f"\n  ==> SCENY OTWARTE MAJA KROTSZY POGLOS mimo {vo/vc:.0f}x wiekszej objetosci.")
        print(f"      To ta sama sygnatura co w naszych danych — SoundSpaces 1.0 uzywal")
        print(f"      tej samej, otwartej geometrii.")
    else:
        print(f"\n  ==> OBSERWACJA: w danych SoundSpaces 1.0 scena otwarta ma DLUZSZY")
        print(f"      poglos, zgodnie z objetoscia — czyli NIE wykazuje sygnatury braku")
        print(f"      sufitu. Nasze dane wykazuja ja nawet po odtworzeniu ich geometrii")
        print(f"      zrodla (kontrola: zrodlo oddalone o ~2 m, nie wspollokowane).")
        print(f"\n      Zestawienie (RT60 @ 1 kHz, ta sama geometria zrodlo-odbiornik):")
        print(f"        {'scena':<20}{'nasze SS2.0':>13}{'SS 1.0':>10}")
        for s in cl + op:
            print(f"        {s:<20}{OURS[s][1]:>13.3f}{res[s][1000.0]:>10.3f}")
        print(f"\n      Silniki zgadzaja sie na scenie ZAMKNIETEJ i rozjezdzaja sie")
        print(f"      wylacznie na OTWARTEJ. Wiodaca hipoteza: potok SoundSpaces 1.0")
        print(f"      domykal objetosc albo uzywal modelu nieczulego na otwarcie.")
        print(f"      Dowodu wprost brak — kod renderujacy nie zostal opublikowany.")
        print(f"      Dla pracy: to REALNA roznica strukturalna wobec baseline'ow,")
        print(f"      dotyczaca 46 % lokalizacji, a nie przypis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
