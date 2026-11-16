#!/usr/bin/env python3
"""Podsumowanie testu Sabine/Eyring z ISTNIEJACYCH logow — bez symulatora i bez GPU.

Odtwarza outputs/measurements/sabine/D_podsumowanie.txt: agregat 8 scen szczelnych,
porownanie z zalatanymi frl_apartment_*, Mann-Whitney (dokladny i asymptotyczny),
kontrole negatywna, czulosc office_4 oraz tabele na scene dla obu regul rzutu V,
tak aby test dalo sie odtworzyc z samego pliku wynikowego.

Wejscie (nic nie jest nadpisywane poza plikiem --out):
    A_sealed.log, C_frl_patched.log          — przebiegi z 2026-07-30
    D_office_4.log, D_office_4_kontrola_office_1.log — przebieg D z 2026-11-16
Zmierzone RT60 ("POZNE", okno -25..-45 dB) czytane z logow z dokladnoscia do
3 miejsc po przecinku; Eyring i Sabine PRZELICZANE z geometrii tym samym kodem
(rt60_vs_sabine.py), wiec obie reguly V sa liczone jednolicie.

Uruchomienie (nie wymaga GPU):
    conda activate habitat
    python my-operations/measurements/rt60_podsumowanie.py            # na stdout
    python my-operations/measurements/rt60_podsumowanie.py --out outputs/measurements/sabine/D_podsumowanie.txt
"""
import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rt60_vs_sabine as r          # noqa: E402  (geometria, Sabine/Eyring — bez zmian)

SABINE_DIR = Path(__file__).resolve().parents[2] / "outputs/measurements/sabine"
PASMA = (500, 1000, 2000)           # pasma agregatu; indeksy w r.BANDS: 2, 3, 4
IDX = {500: 2, 1000: 3, 2000: 4}
SZCZELNE = ("room_0", "office_1", "hotel_0", "apartment_0", "room_1", "room_2", "office_0", "office_4")
ZALATANE = tuple(f"frl_apartment_{i}" for i in range(6))
REGULY = ("max", "ceiling")         # max = regula skryptu; ceiling = analiza wrazliwosci


def czytaj_log(path):
    """-> {scena: {pasmo: (alfa_sr, Sabine, Eyring, pomiar_pozny)}} z tabel w logu."""
    out, cur = {}, None
    for l in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"\[\d+/\d+\] (\w+)", l)
        if m:
            cur = m[1]
            out[cur] = {}
            continue
        m = re.match(r"\s+(125|250|500|1000|2000|4000)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                     r"\s+(\S+)\s+(\S+)\s+\S+", l)
        if m and cur:
            out[cur][int(m[1])] = (float(m[2]), float(m[3]), float(m[4]), float(m[6]))
    return out


def wiersz_pasm(log, scena):
    return [log[scena][b] for b in PASMA]


def geometria(scena, zalatana, regula):
    """-> (V, pokrycie sufitem, Eyring[6], Sabine[6], Eyring_def[6], Sabine_def[6], by_cat, S)"""
    table, default = r.absorption_table()
    by, rzut, h, cr = r.scene_geometry(scena, zalatana)
    foot = rzut if regula == "max" else by.get("ceiling", 0.0)
    V = foot * h
    sab, eyr, S, _ = r.sabine_eyring(by, V, table, default)
    sabd, eyrd, _, _ = r.sabine_eyring(by, V, table, default, force_default=True)
    return V, cr, eyr, sab, eyrd, sabd, by, S


def mw(x, y):
    ex = mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue
    asc = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic",
                       use_continuity=True).pvalue
    asn = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic",
                       use_continuity=False).pvalue
    return ex, asc, asn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="zapisz do pliku zamiast na stdout")
    args = ap.parse_args()

    A = czytaj_log(SABINE_DIR / "A_sealed.log")
    A.update(czytaj_log(SABINE_DIR / "D_office_4.log"))
    C = czytaj_log(SABINE_DIR / "C_frl_patched.log")
    Dk = czytaj_log(SABINE_DIR / "D_office_4_kontrola_office_1.log")
    buf = io.StringIO()
    with redirect_stdout(buf):
        raport(A, C, Dk)
    txt = buf.getvalue()
    if args.out:
        Path(args.out).write_text(txt)
        print(f"zapisano {args.out}")
    else:
        print(txt, end="")


def raport(A, C, Dk):
    print("TEST SABINE/EYRING — przebieg D: office_4 + kontrola odtwarzalnosci (2026-11-16)")
    print("Plik GENEROWANY przez my-operations/measurements/rt60_podsumowanie.py z logow")
    print("A_sealed, C_frl_patched, D_office_4, D_office_4_kontrola_office_1 (bez symulatora).")
    print("Parametry pomiaru: rt60_vs_sabine.py, 30 renderow, rozgrzewka 50, threadCount=1,")
    print("500 promieni, pasma agregatu 500 Hz - 2 kHz. A_sealed.log i summary.txt NIETKNIETE.\n")

    # --- kontrola odtwarzalnosci: office_1 z A vs z D ---
    roz = max(abs(A["office_1"][b][i] - Dk["office_1"][b][i])
              for b in A["office_1"] for i in range(4))
    print("KONTROLA ODTWARZALNOSCI: office_1 (A_sealed.log vs D_office_4_kontrola_office_1.log)")
    print(f"  najwieksza roznica po wszystkich kolumnach tabeli i pasmach 125-4000 Hz: {roz:.3f}\n")

    # --- office_4: geometria ---
    V, cr, eyr, sab, eyrd, sabd, by, S = geometria("office_4", False, "max")
    table, _ = r.absorption_table()
    dflt = sum(a for c, a in by.items() if c not in table)
    print("office_4 (regula V = max):")
    print(f"  V = {V:.1f} m3, S = {S:.1f} m2, pokrycie sufitem {100*cr:.1f} %, "
          f"material domyslny {100*dflt/S:.2f} % ({dflt:.1f} m2)")
    mm = [A["office_4"][b][3] for b in PASMA]
    print(f"  Eyring 500/1k/2k = {', '.join(f'{eyr[IDX[b]]:.3f}' for b in PASMA)} s; "
          f"pomiar = {', '.join(f'{x:.3f}' for x in mm)} s")
    print(f"  zmierzone/Eyring = {', '.join(f'{m/eyr[IDX[b]]:.2f}' for m, b in zip(mm, PASMA))}\n")

    # --- tabela na scene, obie reguly ---
    wyniki = {}
    for regula in REGULY:
        wiersze = []
        for grupa, scen, log, zal in (("szczelne", SZCZELNE, A, False), ("zalatane", ZALATANE, C, True)):
            for s in scen:
                V, cr, eyr, *_ = geometria(s, zal, regula)
                pom = [log[s][b][3] for b in PASMA]
                rat = [p / eyr[IDX[b]] for p, b in zip(pom, PASMA)]
                wiersze.append((grupa, s, V, cr, pom, [eyr[IDX[b]] for b in PASMA], rat,
                                float(np.median(rat))))
        wyniki[regula] = wiersze

    opis = {"max": "REGULA SKRYPTU: rzut = max(ceiling, floor+rug+mat+stair)",
            "ceiling": "ANALIZA WRAZLIWOSCI: rzut = pole `ceiling` (dobrana po obejrzeniu wyniku)"}
    for regula in REGULY:
        print("=" * 100)
        print(f"TABELA NA SCENE — {opis[regula]}")
        print("pom. = zmierzone RT60 'pozne' [s]; Eyr. = Eyring [s]; zm./Eyr. = pom./Eyr.; mediana po 3 pasmach")
        print("=" * 100)
        print(f"{'grupa':<9}{'scena':<17}{'V [m3]':>8}{'sufit%':>7} | "
              f"{'pom.500':>8}{'pom.1k':>8}{'pom.2k':>8} | {'Eyr.500':>8}{'Eyr.1k':>8}{'Eyr.2k':>8} | "
              f"{'r500':>6}{'r1k':>6}{'r2k':>6}{'mediana':>9}")
        for g, s, V, cr, pom, ey, rat, med in wyniki[regula]:
            print(f"{g:<9}{s:<17}{V:>8.1f}{100*cr:>7.1f} | "
                  + "".join(f"{x:>8.3f}" for x in pom) + " | "
                  + "".join(f"{x:>8.3f}" for x in ey) + " | "
                  + "".join(f"{x:>6.2f}" for x in rat) + f"{med:>9.3f}")
        a = [w[7] for w in wyniki[regula] if w[0] == "szczelne"]
        c = [w[7] for w in wyniki[regula] if w[0] == "zalatane"]
        a7 = [w[7] for w in wyniki[regula] if w[0] == "szczelne" and w[1] != "office_4"]
        pts = [x for w in wyniki[regula] if w[0] == "szczelne" for x in w[6]]
        print(f"\n  szczelne n=8: mediana {np.median(a):.3f}x, zakres na scene {min(a):.2f}-{max(a):.2f}x"
              f" | punkty n=24: mediana {np.median(pts):.3f}x, zakres {min(pts):.2f}-{max(pts):.2f}x")
        print(f"  zalatane n=6: mediana {np.median(c):.3f}x, zakres na scene {min(c):.2f}-{max(c):.2f}x")
        print(f"  rozdzielenie zupelne (min szczelnych > max zalatanych): "
              f"{'TAK' if min(a) > max(c) else 'NIE'} ({min(a):.3f} vs {max(c):.3f})")
        for lab, x in (("8 vs 6", a), ("7 vs 6 (bez office_4)", a7)):
            ex, asc, asn = mw(x, c)
            print(f"  Mann-Whitney {lab:<22} dwustronny, poziom sceny: dokladny p = {ex:.4f}; "
                  f"asympt. z poprawka ciaglosci {asc:.4f}, bez {asn:.4f}")
        print()
    print("Test: jednostka = scena (mediana po 3 pasmach), bo pasma tej samej sceny nie sa niezalezne.")
    print("Wartosci do odtworzenia testu: kolumna 'mediana' obu tabel (8 szczelnych i 6 zalatanych).\n")

    # --- blok o regule V ---
    print("=" * 100)
    print("REGULA V — gdzie obie reguly sie roznia")
    print("=" * 100)
    zm = [(w1[1], w1[2], w2[2], w1[3], w1[7], w2[7])
          for w1, w2 in zip(wyniki["max"], wyniki["ceiling"]) if abs(w1[2] - w2[2]) > 1e-6]
    for s, v1, v2, cr, m1, m2 in zm:
        print(f"  {s:<10} pokrycie sufitem {100*cr:.1f} %: V {v1:.1f} -> {v2:.1f} m3 "
              f"(max -> ceiling), mediana zm./Eyring {m1:.2f} -> {m2:.2f}")
    print("  Pozostale 12 scen (6 szczelnych o pokryciu 100 %, 6 zalatanych) — obie reguly identyczne.")
    print("  Wniosek 'zalatane blizej Eyringa niz szczelne' jest spelniony przy obu regulach (p < 0.05);")
    print("  'rozdzielenie zupelne' i 'office_4 jedyna scena ponizej 1.0' zaleza od reguly.\n")

    # --- kontrola negatywna ---
    print("=" * 100)
    print("KONTROLA NEGATYWNA (8 scen, regula V = max): pomiar blizej wariantu REAL czy DEFAULT")
    print("=" * 100)
    lic = {"S": [0, 0], "E": [0, 0]}      # [przy 1 kHz, po wszystkich pasmach]
    for s in SZCZELNE:
        _, _, eyr, sab, eyrd, sabd, _, _ = geometria(s, False, "max")
        lin = []
        for b in PASMA:
            m, i = A[s][b][3], IDX[b]
            rs, re_ = abs(m - sab[i]) < abs(m - sabd[i]), abs(m - eyr[i]) < abs(m - eyrd[i])
            lic["S"][1] += rs
            lic["E"][1] += re_
            if b == 1000:
                lic["S"][0] += rs
                lic["E"][0] += re_
            lin.append(f"{b}:S={'R' if rs else 'D'}/E={'R' if re_ else 'D'}")
        print(f"  {s:<12}" + "  ".join(lin) + f"   (1 kHz: pomiar {A[s][1000][3]:.3f}, "
              f"Sabine real {sab[3]:.3f} / def {sabd[3]:.3f}, Eyring real {eyr[3]:.3f} / def {eyrd[3]:.3f})")
    print(f"  REAL blizej przy 1 kHz: Eyring {lic['E'][0]}/8, Sabine {lic['S'][0]}/8; "
          f"po 3 pasmach: Eyring {lic['E'][1]}/24, Sabine {lic['S'][1]}/24 (R=REAL, D=DEFAULT)\n")

    # --- czulosc office_4 na material domyslny ---
    print("=" * 100)
    print("CZULOSC office_4 NA 9.35 % MATERIALU DOMYSLNEGO (alfa tej czesci = x, regula V = max)")
    print("=" * 100)
    table, default = r.absorption_table()
    by, rzut, h, cr = r.scene_geometry("office_4")
    V = rzut * h
    S = sum(by.values())
    dflt = sum(a for c, a in by.items() if c not in table)
    Aznane = np.zeros(len(r.BANDS))
    for c, a in by.items():
        if c in table:
            Aznane += a * table[c][1]
    mm = np.array([A["office_4"][b][3] for b in PASMA])
    ii = [IDX[b] for b in PASMA]
    print(f"  {'x':>5} | {'Eyring 500/1k/2k [s]':<24} | zm./Eyring 500/1k/2k")
    for x in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75):
        al = (Aznane + dflt * x) / S
        ey = r.SABINE_K * V / (-S * np.log(1 - al))
        tag = "  <- uzyte w tescie" if abs(x - 0.10) < 1e-9 else ""
        print(f"  {x:>5.2f} | {'  '.join(f'{ey[i]:.3f}' for i in ii):<24} | "
              f"{'  '.join(f'{m/ey[i]:.2f}' for m, i in zip(mm, ii))}{tag}")
    print("  Plausibilny zakres x = 0.02-0.30 daje przy 1 kHz zm./Eyring 0.79-1.11.")


if __name__ == "__main__":
    main()
