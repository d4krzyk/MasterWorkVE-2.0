#!/usr/bin/env python3
"""TEST JEDNOSTKOWY: przy N < 8 nadmiarowe rendery sondy sa ODRZUCANE.

Pytanie: sonda robi 8 renderow przy 0 stopni. Gdy regula wyznaczy N < 8, orientacja
0 stopni mialaby wiecej renderow niz pozostale 35 — czyli nizszy szum. Przy N=6 to
sqrt(6/8) = 13 % roznicy. A 0 stopni jest jedna z czterech orientacji bazowych:
w warunku 4-kierunkowym to 1 z 4 probek (25 %), w 36-kierunkowym 1 z 36 (2.8 %).
Sredni poziom szumu roznilby sie wiec systematycznie miedzy warunkami ablacji,
skorelowany DOKLADNIE ze zmienna eksperymentalna. Stad wymog odrzucania nadmiaru.

Po co osobny test: `office_1` (scena walidacyjna) ma N >= 14 we wszystkich
lokalizacjach, wiec ta galaz NIGDY nie wykonuje sie podczas normalnej walidacji.

Metoda: BEZ GPU. Atrapa renderera o szumie tak malym, ze n_raw wychodzi 1, a clamp
podnosi je do N_MIN=6 (czyli < 8). Podmieniamy `echo_core.paths.OUT_ROOT`, wiec
wszystkie sciezki wyjsciowe (wyprowadzane z niej w czasie wywolania) trafiaja do
katalogu tymczasowego i nie dotykaja prawdziwego datasetu.

Wynik: 7/7 kontroli PASS — n_total = 6 (nie 8), n_probe = 6 przy 0 stopni i 0 przy
pozostalych katach, n_planned jednolite, clamped == "min".

Raport: RAPORT_SESJI_2026-07-26_29.md | Dokument: GENERATOR_PARAMS.md §3.3 pkt 2

Uruchomienie (bez GPU):
    python my-operations/measurements/probe_discard_unittest.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import echo_core.paths as paths
from echo_core.params import N_MAX, N_PROBE, TARGET_SNR
from echo_core.store import SPEC_SHAPE


class FakeRenderer:
    """Szum tak maly, ze n_raw wychodzi 1 -> clamp do N_MIN = 6 (czyli < N_PROBE)."""

    def __init__(self, scene, log):
        self.n_renders = 0
        self.n_warmup = 0
        self.render_seconds = 0.0
        self.rng = np.random.default_rng(1234)

    def warmup(self, position, n=None):
        self.n_warmup = 0          # atrapa nie potrzebuje rozgrzewki

    def render(self, position, angle_deg):
        self.n_renders += 1
        spec = np.full(SPEC_SHAPE, 0.1, dtype=np.float32)
        spec = spec + self.rng.normal(0, 1e-3, SPEC_SHAPE).astype(np.float32)
        rgb = np.full((128, 128, 4), 200, dtype=np.uint8)
        depth = np.full((128, 128), 2.0, dtype=np.float32)
        return np.abs(spec), rgb, depth

    def close(self):
        pass


def main():
    tmp = Path(tempfile.mkdtemp(prefix="probe_discard_"))
    real_out = paths.OUT_ROOT
    paths.OUT_ROOT = tmp                      # sciezki licza sie z tej globalnej w czasie wywolania
    try:
        import generate_echo_dataset as gen
        gen.Renderer = FakeRenderer
        gen.build_file_attrs = lambda scene, loc_ids: {
            "scene": scene, "n_locations": len(loc_ids),
            "n_samples_expected": len(loc_ids) * 36,
            "target_snr": TARGET_SNR, "n_max": N_MAX, "n_probe": N_PROBE,
        }
        rc = gen.generate("office_1", limit=2)
        print(f"  kod wyjscia generate(): {rc}")

        with h5py.File(paths.scene_h5("office_1"), "r") as f:
            w = f["written"][:].astype(bool)
            lid = f["location_id"][:][w]
            ang = f["angle_deg"][:][w]
            npl = f["n_planned"][:][w]
            ntot = f["n_total"][:][w]
            npr = f["n_probe"][:][w]
            nraw = f["n_raw"][:][w]
            extra = f["n_rendered_extra"][:][w]
            clamp = np.array([c.decode() for c in f["clamped"][:][w]])

        print(f"\n  probek zapisanych      {len(lid)} (2 lokalizacje x 36)")
        print(f"  n_raw (przed clamp)    {sorted(set(nraw.tolist()))}")
        print(f"  clamped                {sorted(set(clamp.tolist()))}")
        print(f"  n_planned              {sorted(set(npl.tolist()))}")
        print(f"  n_total                {sorted(set(ntot.tolist()))}   <- ma byc [6], NIE [8]")
        print(f"  n_probe przy 0 st.     {sorted(set(npr[ang == 0].tolist()))}")
        print(f"  n_probe przy != 0 st.  {sorted(set(npr[ang != 0].tolist()))}")

        checks = [
            ("n_total == 6 wszedzie (nadmiar sondy odrzucony)", set(ntot.tolist()) == {6}),
            ("n_planned jednolite w lokalizacji",
             all(len(set(npl[lid == l].tolist())) == 1 for l in set(lid.tolist()))),
            ("n_probe == 6 przy 0 st.", set(npr[ang == 0].tolist()) == {6}),
            ("n_probe == 0 przy pozostalych katach", set(npr[ang != 0].tolist()) == {0}),
            ("clamped == 'min'", set(clamp.tolist()) == {"min"}),
            ("brak dorenderowania (snr wysoko)", set(extra.tolist()) == {0}),
            ("komplet 36 katow na lokalizacje",
             all((lid == l).sum() == 36 for l in set(lid.tolist()))),
        ]
        print()
        ok = True
        for name, passed in checks:
            print(f"  [{'OK  ' if passed else 'BLAD'}] {name}")
            ok &= passed
        print(f"\n  WERDYKT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        paths.OUT_ROOT = real_out
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
