#!/usr/bin/env python
"""Zestawienie wynikow sesji 2026-08-15/16 w jeden plik dowodowy. ZERO GPU.

    python my-operations/ml/analysis/final_results.py

Czyta artefakty, ktore zostawily `transfer.py` i `evaluate.py`, i sklada trzy
bloki odpowiadajace sekcjom raportu:

  §2   transfer na OGRANICZONYM zbiorze docelowym (10 % / 25 % / 100 %)
  §3.1 `geometria_echo` na 3 ziarnach -- w tym Delta(B-A) OSOBNO w kazdej geometrii
  §3.2 `glowne` (pelny model) na 3 ziarnach

Wynik: `outputs/ml/echo_ablation/final_results_2026-08-15.json`.

DLACZEGO ISTNIEJE JAKO SKRYPT, a nie jednorazowe wklejenie do konsoli.
`final_results_2026-08-13.json` powstal ad hoc i nie mial skryptu -- przy tej
sesji trzeba bylo odtwarzac recznie, co dokladnie liczyl, zeby rozstrzygnac
sprzecznosc w MAAE (§1.1). Ten plik ma byc uruchamialny ponownie.

DWA TESTY, DWIE ROZNE RZECZY -- nie mylic ich przy pisaniu:
  * test Welcha po ZIARNACH -- czy roznica przezyje ponowne losowanie wag.
    Ma 2-4 stopnie swobody, wiec jest slaby; to jest jego uczciwa cena.
  * bootstrap sparowany po LOKALIZACJACH (`evaluate.py --compare`) -- czy
    roznica przezyje inny zestaw pomieszczen, przy USTALONYCH wagach.
Pierwszy odpowiada na pytanie o istotnosc wyniku, drugi o jego przenoszalnosc.
Raportowane sa oba, bo zadne z osobna nie wystarcza.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402

# Podloga szumu frameworka zmierzona 2026-08-10 §3.1 (dwa przebiegi TEGO SAMEGO
# kodu i tego samego ziarna). Kazda roznica jest do niej odnoszona -- bez tego
# "0,018" nie znaczy nic.
NOISE_FLOOR = (0.0023, 0.0073)

OUT = paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-15.json"


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _finished(run_dir: Path) -> bool:
    """Czy przebieg DOSZEDL DO KONCA.

    Kontrola nieoczywista, ale konieczna: `best.json` powstaje juz przy
    pierwszej walidacji, czyli po ~1 000 z 40 000 krokow. Bez tego strażnika
    skrypt uruchomiony w trakcie kolejki wciaga do sredniej przebieg, ktory
    dopiero sie liczy -- i podaje wynik z n=3, z ktorych jeden jest z 3 %
    budzetu. Dokladnie tak wygladalby blad, ktorego nikt nie zauwazy, bo
    liczba jest prawdopodobna.
    """
    st = _load(run_dir / "status.json")
    return bool(st and st.get("finished"))


def rmse_test36(run_id: str) -> float | None:
    """RMSE na `test@36` z `evaluate.py`. To jest metryka PODSTAWOWA pracy."""
    if not _finished(paths.RUNS_DIR / run_id):
        return None
    d = _load(paths.ML_OUTPUTS / "eval" / run_id / "eval.json")
    if not d:
        return None
    try:
        return float(d["test_sets"]["test@36"]["overall"]["all"]["RMSE"])
    except (KeyError, TypeError):
        return None


def transfer_rmse(label: str, seed: int) -> float | None:
    """RMSE najlepszego kroku zadania docelowego (pelny zbior walidacyjny).

    Zadanie docelowe nie ma warunku w `experiments.py`, wiec nie przechodzi
    przez `evaluate.py` -- liczba pochodzi z `best.json`, tak samo jak WSZYSTKIE
    liczby transferu w raporcie 2026-08-13 §5. Wazne dla spojnosci: to jest ten
    sam pomiar co tam, na tym samym pelnym zbiorze walidacyjnym (6 588 probek),
    NIEZALEZNIE od tego, ile probek treningowych mial przebieg.
    """
    run_dir = paths.ML_OUTPUTS / "pretext_transfer" / f"transfer_{label}_seed{seed}"
    if not _finished(run_dir):
        return None
    d = _load(run_dir / "best.json")
    if not d:
        return None
    try:
        return float(d["all"]["RMSE"])
    except (KeyError, TypeError):
        return None


def _stat(vals) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"values": [], "mean": None, "sd": None, "n": 0}
    return {"values": [float(x) for x in v], "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def _welch(a, b) -> dict:
    """Welch dla `a - b`. p = None przy n < 2 -- z jednego ziarna nie da sie
    orzec o istotnosci, a wypisanie liczby sugerowaloby, ze sie da."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return {"delta": (float(np.mean(a) - np.mean(b)) if a and b else None),
                "p": None, "t": None, "df": None,
                "uwaga": "n < 2 w ktoryms warunku -- brak orzeczenia o istotnosci"}
    t = stats.ttest_ind(a, b, equal_var=False)
    return {"delta": float(np.mean(a) - np.mean(b)), "p": float(t.pvalue),
            "t": float(t.statistic), "df": float(t.df)}


def _vs_floor(delta) -> str | None:
    if delta is None:
        return None
    lo, hi = NOISE_FLOOR
    return f"{abs(delta) / hi:.1f}-{abs(delta) / lo:.1f}x podlogi szumu"


# ------------------------------------------------------------------ §2

LIMITED_INITS = ("scratch", "pretext_K4_seed0", "pretext_K36_seed0")


def sekcja_2_transfer_ograniczony() -> dict:
    """Transfer przy 10 % / 25 % / 100 % zbioru TRENINGOWEGO zadania docelowego.

    Pytanie brzmi: czy Delta wobec `scratch` ROSNIE co do modulu, gdy zbior
    maleje. To, a nie sama wartosc Delty przy 10 %, jest testem diagnozy
    z 2026-08-13 §5.1. Pojedyncza ujemna Delta przy 10 % moglaby byc szumem;
    UPORZADKOWANIE po wielkosci zbioru juz nie.
    """
    plan = {"10%": ("{b}_f10", (0, 1, 2)),
            "25%": ("{b}_f25", (0, 1, 2)),
            # 100 % pochodzi z kolejki 2026-08-13 i ma 5 ziaren, nie 3.
            # Roznica w n jest jawna w wyniku; porownanie miedzy ulamkami
            # dotyczy Delty, a nie liczby ziaren.
            "100%": ("{b}", (0, 1, 2, 3, 4))}
    punkty: dict[str, dict] = {}
    kontrasty: dict[str, dict] = {}
    for pct, (pat, seeds) in plan.items():
        vals = {b: [transfer_rmse(pat.format(b=b), s) for s in seeds] for b in LIMITED_INITS}
        punkty[pct] = {b: _stat(v) for b, v in vals.items()}
        for b in LIMITED_INITS:
            if b == "scratch":
                continue
            w = _welch(vals[b], vals["scratch"])
            w["krotnosc_podlogi_szumu"] = _vs_floor(w["delta"])
            w["kierunek"] = (None if w["delta"] is None else
                             ("pretrening LEPSZY" if w["delta"] < 0 else "pretrening GORSZY"))
            kontrasty[f"{b}_vs_scratch@{pct}"] = w

    # WERDYKT, A NIE SAMO UPORZADKOWANIE.
    #
    # Kusi, zeby uznac przewidywanie za potwierdzone, gdy Delta ustawia sie
    # monotonicznie. To bylby blad: trzy roznice, z ktorych zadna nie jest
    # istotna, ustawiaja sie monotonicznie z prawdopodobienstwem 1/6 przez sam
    # przypadek. Przewidywanie z §5.1 brzmi "pretrening ZACZNIE POMAGAC" --
    # zeby je potwierdzic, przy 10 % zbioru pretrening musi byc LEPSZY od
    # `scratch`, i to o wiecej niz podloga szumu. Uporzadkowanie jest warunkiem
    # dodatkowym, nie wystarczajacym.
    trend = {}
    for b in LIMITED_INITS:
        if b == "scratch":
            continue
        d = {pct: kontrasty.get(f"{b}_vs_scratch@{pct}", {}).get("delta")
             for pct in ("10%", "25%", "100%")}
        p10 = kontrasty.get(f"{b}_vs_scratch@10%", {}).get("p")
        pelne = all(d[p] is not None for p in d)
        mono = bool(d["10%"] < d["25%"] < d["100%"]) if pelne else None
        pomaga = (d["10%"] is not None and d["10%"] < -NOISE_FLOOR[1])
        istotny = (p10 is not None and p10 < 0.05)
        if not pelne:
            werdykt = "NIEKOMPLETNE"
        elif pomaga and istotny:
            werdykt = "POTWIERDZONE"
        elif pomaga:
            werdykt = "SLABE POPARCIE (kierunek zgodny, ale nieistotny)"
        else:
            werdykt = "OBALONE"
        trend[b] = {
            "delta_wg_ulamka": d,
            "uporzadkowanie_monotoniczne": mono,
            "pretrening_lepszy_przy_10pct_ponad_szumem": pomaga,
            "istotny_przy_10pct": istotny,
            "werdykt": werdykt,
            "przewidywanie_5_1": "przy 10 % zbioru pretrening ma byc LEPSZY od `scratch` "
                                 "o wiecej niz podloga szumu (0,0073); samo uporzadkowanie "
                                 "trzech nieistotnych roznic nie wystarcza",
        }
    return {"opis": "zadanie docelowe RGB2Depth bez audio; ograniczany WYLACZNIE zbior "
                    "treningowy, walidacja i test pelne; podzbior stratyfikowany po "
                    "lokalizacji, ziarno podzbioru 20260815 STALE miedzy warunkami "
                    "i ziarnami sieci; budzet 40 000 krokow we wszystkich warunkach",
            "n_probek_treningowych": {"10%": 4946, "25%": 12366, "100%": 49464},
            "rownowaznik_epok": {"10%": 258.8, "25%": 103.5, "100%": 25.9},
            "punkty": punkty, "kontrasty": kontrasty, "trend": trend}


# ---------------------------------------------------------------- §3.1


def sekcja_31_geometria_echo() -> dict:
    """`geometria_echo` na 3 ziarnach + Delta(B-A) OSOBNO w kazdej geometrii.

    DLACZEGO Delta(B-A), A NIE SUROWE RMSE. Modele `main` i `patched` sa
    punktowane na roznych zbiorach pikseli waznych (latka domyka dziury w
    glebi), wiec ich surowe RMSE nie sa wprost porownywalne -- to jest cale
    zastrzezenie o masce z 2026-08-11 §5. Efekt GESTOSCI KATOWEJ jest natomiast
    roznica WEWNATRZ jednej geometrii, wiec wybor maski skraca sie w
    odejmowaniu. Pytanie "czy efekt gestosci zachowuje sie w obu geometriach"
    jest wiec pytaniem o Delta(B-A) w `main` wobec Delta(B-A) w `patched`,
    a nie o RMSE(EPB) wobec RMSE(EB).
    """
    seeds = (0, 1, 2)
    par = {"main": {"A": "EA", "B": "EB", "D": "ED"},
           "patched": {"A": "EPA", "B": "EPB", "D": "EPD"}}
    punkty: dict[str, dict] = {}
    per_seed: dict[str, dict[str, list]] = {}
    for geo, m in par.items():
        punkty[geo], per_seed[geo] = {}, {}
        for rola, cond in m.items():
            v = [rmse_test36(f"{cond}_seed{s}") for s in seeds]
            punkty[geo][cond] = _stat(v)
            per_seed[geo][rola] = v

    # Delta liczona PER ZIARNO, a nie jako roznica srednich: ziarno s w `main`
    # i w `patched` startuje z tej samej inicjalizacji wag, wiec parowanie po
    # ziarnie usuwa czesc rozrzutu inicjalizacji z obu stron naraz.
    def _delta(geo, x, y):
        return [(a - b) if (a is not None and b is not None) else None
                for a, b in zip(per_seed[geo][x], per_seed[geo][y])]

    efekt: dict[str, dict] = {}
    for lab, (x, y) in {"gestosc_D_minus_A": ("D", "A"),
                        "laczny_B_minus_A": ("B", "A"),
                        "ilosc_danych_B_minus_D": ("B", "D")}.items():
        dm, dp = _delta("main", x, y), _delta("patched", x, y)
        dmv = [q for q in dm if q is not None]
        dpv = [q for q in dp if q is not None]
        efekt[lab] = {
            "main": _stat(dm), "patched": _stat(dp),
            "patched_minus_main": _welch(dp, dm),
            "znak_zgodny": (bool(np.sign(np.mean(dmv)) == np.sign(np.mean(dpv)))
                            if dmv and dpv else None),
        }

    geo_delta = {}
    for rola in ("A", "B", "D"):
        k = f"{par['patched'][rola]}_minus_{par['main'][rola]}"
        geo_delta[k] = {**_welch(per_seed["patched"][rola], per_seed["main"][rola]),
                        "uwaga": "wartosc DODATNIA = `patched` GORSZY; maska pelna"}
        geo_delta[k]["krotnosc_podlogi_szumu"] = _vs_floor(geo_delta[k]["delta"])

    return {"opis": "echo2depth, 3 ziarna, RMSE test@36, maska pelna",
            "punkty": punkty,
            "patched_minus_main": geo_delta,
            "efekt_gestosci_w_obu_geometriach": efekt}


# ---------------------------------------------------------------- §3.2


def sekcja_32_glowne() -> dict:
    """Pelny model A/B/D na 3 ziarnach -- domkniecie istotnosci §2 z 2026-08-13."""
    seeds = (0, 1, 2)
    vals = {c: [rmse_test36(f"{c}_seed{s}") for s in seeds] for c in ("A", "B", "D")}
    punkty = {c: _stat(v) for c, v in vals.items()}

    def _paired(x, y):
        """Test SPAROWANY po ziarnie, nie Welch dla dwoch prob niezaleznych.

        Warunki A, B i D z tym samym ziarnem maja te sama inicjalizacje wag,
        wiec roznica per ziarno ma mniejszy rozrzut niz roznica srednich.
        Przy trzech ziarnach to jest roznica miedzy orzeczeniem a jego brakiem.
        """
        d = [(a - b) if (a is not None and b is not None) else None
             for a, b in zip(vals[x], vals[y])]
        st = _stat(d)
        dd = [q for q in d if q is not None]
        if len(dd) > 1:
            t = stats.ttest_1samp(dd, 0.0)
            st["p_sparowany"] = float(t.pvalue)
            st["t"] = float(t.statistic)
        else:
            st["p_sparowany"] = None
        st["krotnosc_podlogi_szumu"] = _vs_floor(st["mean"])
        return st

    skladowe = {"gestosc_D_minus_A": _paired("D", "A"),
                "ilosc_danych_B_minus_D": _paired("B", "D"),
                "laczny_B_minus_A": _paired("B", "A")}
    g = skladowe["gestosc_D_minus_A"]["mean"]
    l = skladowe["laczny_B_minus_A"]["mean"]
    ud = (100.0 * g / l) if (g is not None and l not in (None, 0)) else None

    boot = {}
    for a, b in (("A", "D"), ("A", "B")):
        for s in seeds:
            d = _load(paths.ML_OUTPUTS / "eval" / f"compare_{a}_seed{s}_vs_{b}_seed{s}.json")
            if d:
                boot[f"{a}_vs_{b}_seed{s}"] = {
                    k: d["all"][k] for k in ("delta_point", "ci_low", "ci_high",
                                             "ci_excludes_zero", "n_locations")}

    return {"opis": "pelny model (RGB + echo + material + uwaga), 3 ziarna, RMSE test@36",
            "punkty": punkty, "skladowe": skladowe,
            "udzial_gestosci_pct": ud,
            "bootstrap_po_lokalizacjach": boot,
            "przedrejestrowana_regula": {
                "data": "2026-08-11 §2", "bound_przewidziany": 0.00529, "prog": 0.015,
                "decyzja": "1 ziarno",
                "co_sie_okazalo": "zmierzone D-A = 0,01831, czyli 3,46x wiecej niz bound; "
                                  "ziarna 1-2 dolozone POST HOC 2026-08-15 §3.2"}}


def main(argv=None) -> int:
    payload = {
        "opis": "Wyniki sesji 2026-08-15/16: transfer na ograniczonym zbiorze docelowym "
                "oraz domkniecie istotnosci `geometria_echo` i `glowne`.",
        "podloga_szumu": NOISE_FLOOR,
        "transfer_ograniczony": sekcja_2_transfer_ograniczony(),
        "geometria_echo_3ziarna": sekcja_31_geometria_echo(),
        "glowne_3ziarna": sekcja_32_glowne(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")

    t = payload["transfer_ograniczony"]
    print("=" * 90)
    print("§2  TRANSFER NA OGRANICZONYM ZBIORZE DOCELOWYM")
    print("=" * 90)
    print(f"  {'ulamek':>7s} {'warunek':24s} {'ziarn':>5s} {'RMSE':>9s} {'sd':>9s} "
          f"{'d vs scratch':>13s} {'p':>7s}")
    for pct in ("10%", "25%", "100%"):
        for b, st in t["punkty"][pct].items():
            if not st["n"]:
                continue
            k = t["kontrasty"].get(f"{b}_vs_scratch@{pct}", {})
            d = "—" if k.get("delta") is None else f"{k['delta']:+.5f}"
            p = "—" if k.get("p") is None else f"{k['p']:.3f}"
            sd = "  n<2" if st["sd"] is None else f"{st['sd']:.5f}"
            print(f"  {pct:>7s} {b:24s} {st['n']:>5d} {st['mean']:>9.5f} {sd:>9s} "
                  f"{d:>13s} {p:>7s}")
    print("\n  TREND (§5.1 przewiduje: przy 10 % pretrening LEPSZY od scratch ponad szumem)")
    for b, tr in t["trend"].items():
        d = tr["delta_wg_ulamka"]
        row = "   ".join(f"{p}: {('—' if d[p] is None else f'{d[p]:+.5f}')}"
                         for p in ("10%", "25%", "100%"))
        print(f"    {b:22s} {row}")
        print(f"    {'':22s} monotoniczne={tr['uporzadkowanie_monotoniczne']}  "
              f"lepszy@10%={tr['pretrening_lepszy_przy_10pct_ponad_szumem']}  "
              f"istotny={tr['istotny_przy_10pct']}  ->  {tr['werdykt']}")

    g = payload["geometria_echo_3ziarna"]
    print("\n" + "=" * 90)
    print("§3.1  GEOMETRIA_ECHO NA 3 ZIARNACH")
    print("=" * 90)
    for geo, m in g["punkty"].items():
        for c, st in m.items():
            if st["n"]:
                sd = "  n<2" if st["sd"] is None else f"{st['sd']:.5f}"
                print(f"  {geo:8s} {c:5s} n={st['n']}  {st['mean']:.5f} +/- {sd}")
    print("\n  patched - main (wartosc dodatnia = `patched` gorszy):")
    for k, v in g["patched_minus_main"].items():
        if v["delta"] is not None:
            p = "—" if v["p"] is None else f"{v['p']:.4f}"
            print(f"    {k:22s} {v['delta']:+.5f}  p={p}  {v['krotnosc_podlogi_szumu']}")
    print("\n  EFEKT GESTOSCI OSOBNO W KAZDEJ GEOMETRII (to jest wielkosc porownywana):")
    for lab, e in g["efekt_gestosci_w_obu_geometriach"].items():
        mm, pp = e["main"], e["patched"]
        if mm["n"] and pp["n"]:
            pm = e["patched_minus_main"]
            p = "—" if pm["p"] is None else f"{pm['p']:.3f}"
            print(f"    {lab:24s} main {mm['mean']:+.5f}   patched {pp['mean']:+.5f}   "
                  f"roznica {pm['delta']:+.5f} (p={p})  znak_zgodny={e['znak_zgodny']}")

    gl = payload["glowne_3ziarna"]
    print("\n" + "=" * 90)
    print("§3.2  GLOWNE (PELNY MODEL) NA 3 ZIARNACH")
    print("=" * 90)
    for c, st in gl["punkty"].items():
        if st["n"]:
            sd = "  n<2" if st["sd"] is None else f"{st['sd']:.5f}"
            print(f"  {c:3s} n={st['n']}  {st['mean']:.5f} +/- {sd}")
    for lab, st in gl["skladowe"].items():
        if st["n"]:
            p = "—" if st.get("p_sparowany") is None else f"{st['p_sparowany']:.4f}"
            print(f"  {lab:26s} {st['mean']:+.5f}  p_sparowany={p}  "
                  f"{st['krotnosc_podlogi_szumu']}")
    if gl["udzial_gestosci_pct"] is not None:
        print(f"  udzial gestosci w efekcie lacznym: {gl['udzial_gestosci_pct']:.1f} %")
    print(f"\nzapisano: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
