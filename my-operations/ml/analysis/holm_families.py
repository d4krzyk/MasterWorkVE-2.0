#!/usr/bin/env python
"""Poprawka Holma (Holm–Bonferroni, step-down) W OBREBIE rodzin testow. Zero GPU.

    python my-operations/ml/analysis/holm_families.py [--out DIR] [--dry-run]

p SUROWE sa liczone dokladnie tak, jak w `echo_only_tables.py` / `final_results.py`
(te same funkcje sa stamtad IMPORTOWANE, a nie przepisywane -- druga implementacja
mogla by sie po cichu rozjechac z liczbami w pracy):

  (a) siec tylko-echo, D−A / B−D / B−A: `ttest_1samp` na roznicach per ziarno
      (ED−EA, EB−ED, EB−EA), test@36, warstwa `all`, 3 ziarna;
  (b) krzywa stalego budzetu: EK6/EK9/EK12/EK18 − ED (K=36), Welch 3 vs 3;
  (c) model pelny, D−A / B−D / B−A: `ttest_1samp` na roznicach per ziarno
      (`final_results.sekcja_32_glowne`, ze straznikiem `status.json finished`);
  (d) geometria: EPA−EA, EPD−ED, EPB−EB: Welch na RMSE test@36, maska pelna
      (`final_results.sekcja_31_geometria_echo()['patched_minus_main']`);
  (e) transfer: 4 x pretrening − scratch, 5 ziaren transferu, RMSE z `best.json`,
      Welch (`final_results.transfer_rmse`, ktory sprawdza `status.json finished`).

Holm liczony jest W RODZINIE (to jest wynik podstawowy). Holm po WSZYSTKICH
testach razem jest podany osobno i opisany jako DODATKOWY -- rodziny odpowiadaja
osobnym pytaniom badawczym, wiec laczna korekta jest kontrola konserwatywna,
a nie wlasciwa jednostka wnioskowania.

Wyniki: `<DIR>/holm_families.json` i `<DIR>/holm_families.md`; przy `--dry-run`
do `<DIR>/dryrun/`. Skrypt NIE nadpisuje: jesli ktorys plik wynikowy istnieje,
konczy sie bledem.

Na koncu: KONTROLA p surowych wobec liczb z dokumentacji (tylko wypisuje
niezgodnosci, niczego nie poprawia).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402

DEFAULT_OUT = paths.ML_OUTPUTS / "overnight_2026-11"
ALPHA = 0.05
TRANSFER_SEEDS = (0, 1, 2, 3, 4)
TRANSFER_LABELS = ("pretext_K4_seed0", "pretext_K36_p16_seed0", "pretext_K36_seed0",
                   "pretext_K12_seed0")


# ------------------------------------------------------------------ Holm

def holm(pvals) -> list[float]:
    """Holm–Bonferroni step-down, p skorygowane (monotoniczne, obciete do 1).

    p_(i) posortowane rosnaco; p_holm_(i) = max_{j<=i} min(1, (m − j + 1) · p_(j)).
    Wlasna implementacja (kilka linii), zeby nie zalezec od obecnosci statsmodels;
    jesli statsmodels jest zainstalowany, wynik jest z nim porownywany w `main`.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=float)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return [float(x) for x in adj]


def _sprawdz_statsmodels(pvals, adj) -> str:
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return "statsmodels niedostepny -- tylko wlasna implementacja"
    ref = multipletests(pvals, alpha=ALPHA, method="holm")[1]
    ok = bool(np.allclose(ref, adj, rtol=0, atol=1e-12))
    return f"zgodne ze statsmodels.multipletests(method='holm'): {ok}"


# ------------------------------------------------------------------ rodziny

def rodzina_a() -> list[dict]:
    from .echo_only_tables import paired, rmse
    R = {c: rmse(c) for c in ("EA", "ED", "EB")}
    out = []
    for lab, (x, y) in {"D−A (ED−EA)": ("ED", "EA"), "B−D (EB−ED)": ("EB", "ED"),
                        "B−A (EB−EA)": ("EB", "EA")}.items():
        r = paired(R[x], R[y])
        out.append({"test": lab, "delta": r["mean"], "p_raw": r["p"],
                    "metoda": "ttest_1samp na roznicach per ziarno (df=2)", "n": [3, 3]})
    return out


def rodzina_b() -> list[dict]:
    from .echo_only_tables import rmse, welch
    r36 = rmse("ED")
    out = []
    for c, k in (("EK6", 6), ("EK9", 9), ("EK12", 12), ("EK18", 18)):
        w = welch(rmse(c), r36)
        out.append({"test": f"K={k} − K=36 ({c}−ED)", "delta": w["delta"], "p_raw": w["p"],
                    "metoda": f"Welch 3 vs 3 (df={w['df']:.2f})", "n": [3, 3]})
    return out


def rodzina_c() -> list[dict]:
    from .final_results import sekcja_32_glowne
    sk = sekcja_32_glowne()["skladowe"]
    out = []
    for lab, k in (("D−A", "gestosc_D_minus_A"), ("B−D", "ilosc_danych_B_minus_D"),
                   ("B−A", "laczny_B_minus_A")):
        s = sk[k]
        out.append({"test": lab, "delta": s["mean"], "p_raw": s.get("p_sparowany"),
                    "metoda": "ttest_1samp na roznicach per ziarno (df=n−1)",
                    "n": [s["n"], s["n"]]})
    return out


def rodzina_d() -> list[dict]:
    from .final_results import sekcja_31_geometria_echo
    g = sekcja_31_geometria_echo()
    pm, pk = g["patched_minus_main"], g["punkty"]
    out = []
    for p, m in (("EPA", "EA"), ("EPD", "ED"), ("EPB", "EB")):
        w = pm[f"{p}_minus_{m}"]
        out.append({"test": f"{p}−{m}", "delta": w["delta"], "p_raw": w["p"],
                    "metoda": "Welch na RMSE test@36, maska pelna"
                              + ("" if w.get("df") is None else f" (df={w['df']:.2f})"),
                    "n": [pk["patched"][p]["n"], pk["main"][m]["n"]]})
    return out


def rodzina_e() -> tuple[list[dict], dict]:
    from .final_results import _welch, transfer_rmse
    status = {}
    vals = {}
    for lab in ("scratch",) + TRANSFER_LABELS:
        v = [transfer_rmse(lab, s) for s in TRANSFER_SEEDS]   # None = brak finished/best.json
        vals[lab] = v
        status[lab] = {f"seed{s}": (x is not None) for s, x in zip(TRANSFER_SEEDS, v)}
    out = []
    for lab in TRANSFER_LABELS:
        w = _welch(vals[lab], vals["scratch"])
        out.append({"test": f"{lab} − scratch", "delta": w["delta"], "p_raw": w["p"],
                    "metoda": "Welch na RMSE best.json['all'] (pelna walidacja)"
                              + ("" if w.get("df") is None else f" (df={w['df']:.2f})"),
                    "n": [sum(x is not None for x in vals[lab]),
                          sum(x is not None for x in vals["scratch"])]})
    return out, status


ROD_OPIS = {
    "a": "siec tylko-echo (echo2depth): kontrasty D−A, B−D, B−A, test@36, 3 ziarna",
    "b": "krzywa stalego budzetu (echo2depth): K − K36, Welch 3 vs 3",
    "c": "model pelny (RGB+echo+material+uwaga): D−A, B−D, B−A, 3 ziarna",
    "d": "geometria: patched − main, Welch, maska pelna, 3 ziarna",
    "e": "transfer pretrening vs scratch, 5 ziaren, RMSE best.json",
}


def _dopisz_holm(testy: list[dict], klucz: str) -> None:
    ok = [i for i, t in enumerate(testy) if t["p_raw"] is not None]
    adj = holm([testy[i]["p_raw"] for i in ok]) if ok else []
    for t in testy:
        t[klucz] = None
        t[klucz + "_lt_0.05"] = None
    for i, a in zip(ok, adj):
        testy[i][klucz] = a
        testy[i][klucz + "_lt_0.05"] = bool(a < ALPHA)


# ------------------------------------------------------------------ kontrola

KONTROLA = [  # (rodzina, test, oczekiwane p surowe, tolerancja = polowa ostatniej cyfry)
    ("a", "D−A (ED−EA)", 0.0026, 5e-5),
    ("a", "B−D (EB−ED)", 0.0016, 5e-5),
    ("a", "B−A (EB−EA)", 0.0010, 5e-5),
    ("b", "K=9 − K=36 (EK9−ED)", 0.0032, 5e-5),
    ("c", "D−A", 0.0096, 5e-5),
    ("c", "B−D", 0.0010, 5e-5),
    ("c", "B−A", 0.0018, 5e-5),
    ("d", "EPB−EB", 0.013, 5e-4),
]


def kontrola(rodziny: dict) -> list[dict]:
    out = []
    for r, t, e, tol in KONTROLA:
        rec = next((x for x in rodziny[r]["testy"] if x["test"] == t), None)
        o = None if rec is None else rec["p_raw"]
        out.append({"rodzina": r, "test": t, "oczekiwane": e, "obliczone": o, "tolerancja": tol,
                    "zgodne": bool(o is not None and abs(o - e) <= tol)})
    return out


# ------------------------------------------------------------------ render

def _p(p) -> str:
    if p is None:
        return "—"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def render_md(P: dict) -> str:
    L = ["# Poprawka Holma w rodzinach testów", "",
         "Generowane przez `my-operations/ml/analysis/holm_families.py`. Nie edytować ręcznie.",
         "Konwencja: `X − Y` = dosłownie X − Y (ujemne = mniejszy błąd = lepiej). RMSE głębi [m]. "
         "Holm–Bonferroni step-down, α = 0,05.", ""]
    if P["dry_run"]:
        L += ["**DRY-RUN** — przebieg kontrolny, liczby te same co w pełnym (skrypt nie ma "
              "parametrów przyspieszających), zapisany do podkatalogu `dryrun/`.", ""]
    L += ["## Holm w obrębie rodzin (wynik podstawowy)", ""]
    for r, R in P["rodziny"].items():
        L += [f"### ({r}) {R['opis']}", "", f"Liczba testów w rodzinie: **{R['m']}**.", "",
              "| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |",
              "|---|---|---|---|---|---|---|"]
        for t in R["testy"]:
            d = "—" if t["delta"] is None else f"{t['delta']:+.5f}"
            sig = "—" if t["p_holm_rodzina_lt_0.05"] is None else (
                "tak" if t["p_holm_rodzina_lt_0.05"] else "nie")
            L.append(f"| {t['test']} | {d} | {_p(t['p_raw'])} | {_p(t['p_holm_rodzina'])} | {sig} | "
                     f"{t['n'][0]} vs {t['n'][1]} | {t['metoda']} |")
        L.append("")
    L += ["Status przebiegów transferu (`status.json` finished i obecny `best.json`):", ""]
    for lab, st in P["rodziny"]["e"]["status_finished"].items():
        L.append(f"- {lab}: " + ", ".join(f"{k}={'tak' if v else 'NIE'}" for k, v in st.items()))
    L += ["", "## DODATKOWO: Holm po wszystkich testach razem", "",
          f"Kontrola konserwatywna (m = {P['wszystkie']['m']}); nie zastępuje korekty w rodzinach.", "",
          "| rodzina | test | p surowe | p Holm (wszystkie) | p_Holm < 0,05 |", "|---|---|---|---|---|"]
    for t in P["wszystkie"]["testy"]:
        sig = "—" if t["p_holm_wszystkie_lt_0.05"] is None else (
            "tak" if t["p_holm_wszystkie_lt_0.05"] else "nie")
        L.append(f"| {t['rodzina']} | {t['test']} | {_p(t['p_raw'])} | {_p(t['p_holm_wszystkie'])} | {sig} |")
    L += ["", f"Implementacja Holma: {P['holm_sprawdzenie']}.", "",
          "## Kontrola p surowych wobec dokumentacji", "",
          "| rodzina | test | oczekiwane | obliczone | |", "|---|---|---|---|---|"]
    for k in P["kontrola"]:
        o = "—" if k["obliczone"] is None else f"{k['obliczone']:.5f}"
        L.append(f"| {k['rodzina']} | {k['test']} | {k['oczekiwane']} | {o} | "
                 f"{'zgodne' if k['zgodne'] else '**NIEZGODNE**'} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"katalog wynikow (domyslnie {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="przebieg kontrolny do <DIR>/dryrun/ (ten sam rachunek -- jest szybki)")
    args = ap.parse_args(argv)
    t0 = time.time()

    out_dir = args.out / "dryrun" if args.dry_run else args.out
    f_json, f_md = out_dir / "holm_families.json", out_dir / "holm_families.md"
    istnieje = [str(p) for p in (f_json, f_md) if p.exists()]
    if istnieje:
        print("BLAD: plik(i) wynikowe juz istnieja, nie nadpisuje:\n  " + "\n  ".join(istnieje),
              file=sys.stderr)
        return 2

    e_testy, e_status = rodzina_e()
    rodziny = {"a": {"testy": rodzina_a()}, "b": {"testy": rodzina_b()},
               "c": {"testy": rodzina_c()}, "d": {"testy": rodzina_d()},
               "e": {"testy": e_testy, "status_finished": e_status}}
    for r, R in rodziny.items():
        R["opis"] = ROD_OPIS[r]
        R["m"] = sum(t["p_raw"] is not None for t in R["testy"])
        _dopisz_holm(R["testy"], "p_holm_rodzina")
    rodziny = {r: {"opis": R["opis"], "m": R["m"], **{k: v for k, v in R.items()
                                                     if k not in ("opis", "m")}}
               for r, R in rodziny.items()}

    wszystkie = [{"rodzina": r, "test": t["test"], "p_raw": t["p_raw"]}
                 for r, R in rodziny.items() for t in R["testy"]]
    _dopisz_holm(wszystkie, "p_holm_wszystkie")
    pv = [t["p_raw"] for t in wszystkie if t["p_raw"] is not None]
    spr = _sprawdz_statsmodels(pv, holm(pv))

    P = {"opis": "Holm–Bonferroni step-down w obrebie rodzin; dodatkowo po wszystkich testach",
         "alpha": ALPHA, "dry_run": bool(args.dry_run),
         "rodziny": rodziny,
         "wszystkie": {"opis": "DODATKOWE: Holm po wszystkich testach razem (konserwatywne)",
                       "m": len(pv), "testy": wszystkie},
         "holm_sprawdzenie": spr}
    P["kontrola"] = kontrola(rodziny)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(f_json, "x", encoding="utf-8") as fh:        # "x" = blad, jesli plik istnieje
        fh.write(json.dumps(P, indent=2, ensure_ascii=False))
    with open(f_md, "x", encoding="utf-8") as fh:
        fh.write(render_md(P))

    print(f"zapisano: {f_json}\n          {f_md}")
    for r, R in rodziny.items():
        print(f"({r}) m={R['m']}: " + "; ".join(
            f"{t['test']} p={_p(t['p_raw'])} holm={_p(t['p_holm_rodzina'])}" for t in R["testy"]))
    print(spr)
    print("KONTROLA p surowych:")
    for k in P["kontrola"]:
        o = "—" if k["obliczone"] is None else f"{k['obliczone']:.5f}"
        print(f"  [{'OK ' if k['zgodne'] else 'NIEZGODNE'}] ({k['rodzina']}) {k['test']:<22s} "
              f"oczekiwane {k['oczekiwane']}  obliczone {o}")
    print(f"czas: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
