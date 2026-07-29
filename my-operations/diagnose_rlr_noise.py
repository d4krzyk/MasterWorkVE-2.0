#!/usr/bin/env python3
"""Diagnostyka szumu i wlasciwosci RLRAudioPropagation (Visual Echoes 2.0).

Uruchomienie (env `habitat` aktywne):

    conda activate habitat
    python my-operations/diagnose_rlr_noise.py --exp p0
    python my-operations/diagnose_rlr_noise.py --exp noise_floor_scenes noise_floor_remaining

Ten plik to samo CLI. Eksperymenty mieszkaja w pakiecie `diagnostics/`, po jednym
module na temat:

    common            sciezki, budowa Simulatora, pozycje, render, estymatory
    exp_determinism   E1 + checkpoint-boundary (sekwencja RNG, wznawianie)
    exp_rays          E2 (liczba promieni, bias katowy, budzet watkow)
    exp_averaging     E3 (domena usredniania), E4 (dlugosc IR)
    exp_materials     weryfikacja configu materialow Repliki
    exp_noise_floor   podloga szumu, wysokosc sluchacza, census scen
    exp_gpu           skalowanie pamieci GPU/RSS

Wspolny raport JSON (diagnostics_report.json) jest SCALANY po kazdym uruchomieniu,
wiec kolejne sesje dopisuja wyniki bez nadpisywania wczesniejszych. Klucze rejestru
sa jednoczesnie kluczami w raporcie — nie zmieniac ich nazw.
"""

import argparse
import json

from diagnostics import EXPERIMENTS
from diagnostics.common import OUT_DIR, REPORT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=sorted(EXPERIMENTS.keys()),
        required=True,
        help="Ktore eksperymenty uruchomic (mozna kilka na raz).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report = json.load(f)

    for name in args.exp:
        report[name] = EXPERIMENTS[name]()

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nRaport zapisany w: {REPORT_PATH}")
    print("\n=== PODSUMOWANIE ===")
    for name in args.exp:
        status = report[name].get("status", "-")
        print(f"  {name}: status={status}")


if __name__ == "__main__":
    main()
