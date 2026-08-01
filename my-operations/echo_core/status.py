"""--status: tabelka stanu wszystkich 18 scen. Bez GPU."""

import json
import os
import time

import numpy as np

from .params import MEAN_N_SPEC, N_ANGLES
from . import paths
from .paths import scene_h5, scene_progress
from .runtime import _fmt_hms
from .scenes import HELD_OUT, load_scene_locations, scenes_for_variant
from .verify import _open_readonly

def _writer_alive(scene):
    """Czy proces, ktory zapisal sidecar tej sceny, wciaz zyje.

    os.kill(pid, 0) nie wysyla sygnalu — tylko sprawdza istnienie procesu
    i uprawnienia. Ryzyko recyklingu PID-u jest tu bez znaczenia: najgorszy
    skutek to jeden zle opisany wiersz tabelki.
    """
    sp = scene_progress(scene)
    if not sp.exists():
        return False
    try:
        pid = int(json.loads(sp.read_text(encoding="utf-8")).get("pid", -1))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def status():
    print(f"\n{'=' * 96}")
    print(f"  STAN DATASETU [{paths.VARIANT}] — {paths.OUT_ROOT}")
    print(f"  kolejnosc wg GENERATOR_PARAMS.md §4.2 (walidacyjna, held-out, po jednej z rodziny, reszta)")
    print("=" * 96)
    print(f"  {'#':<3}{'scena':<18}{'':<3}{'stan':<12}{'probki':>14}{'%':>7}{'rozmiar':>11}"
          f"{'s/render':>10}{'czas':>10}")
    print("  " + "-" * 92)

    tot_done = tot_expected = 0
    tot_bytes = 0
    tot_seconds = 0.0
    rates = []
    for i, scene in enumerate(scenes_for_variant(), start=1):
        path = scene_h5(scene)
        tag = "H" if scene in HELD_OUT else " "
        n_written = n_expected = 0
        size = 0
        spr = 0.0
        secs = 0.0
        state = "brak"

        if path.exists():
            size = path.stat().st_size
            info = None
            try:
                f = _open_readonly(path)
                try:
                    w = f["written"][:]
                    n_written = int(w.sum())
                    n_expected = int(f.attrs.get("n_samples_expected", w.size))
                    spr = float(f.attrs.get("seconds_per_render", 0.0))
                    secs = sum(r.get("wall_seconds", 0.0)
                               for r in json.loads(f.attrs.get("runs", "[]")))
                finally:
                    f.close()
                info = "h5"
            except Exception:
                # Plik trzymany do zapisu przez inny proces — sidecar JSON.
                sp = scene_progress(scene)
                if sp.exists():
                    d = json.loads(sp.read_text(encoding="utf-8"))
                    n_written = int(d.get("n_written", 0))
                    n_expected = int(d.get("n_samples_expected", 0))
                    spr = float(d.get("seconds_per_render", 0.0) or 0.0)
                    info = "sidecar"
            if info is None:
                state = "NIECZYTELNY"
            elif n_expected and n_written >= n_expected:
                state = "gotowa"
            else:
                # Zywotnosc procesu, a nie mtime sidecara, jest tu sygnalem
                # rozstrzygajacym: jedna lokalizacja przy N=29 trwa ~5 min, wiec
                # kazdy prog oparty na czasie mylilby "wolno" z "przerwane".
                state = "przerwana" if n_written else "pusta"
                if _writer_alive(scene):
                    state = "W TOKU"
        else:
            try:
                n_expected = len(load_scene_locations(scene)[0]) * N_ANGLES
            except Exception:
                n_expected = 0

        tot_done += n_written
        tot_expected += n_expected
        tot_bytes += size
        tot_seconds += secs
        if spr > 0:
            rates.append(spr)
        pct = 100.0 * n_written / n_expected if n_expected else 0.0
        print(f"  {i:<3}{scene:<18}{tag:<3}{state:<12}{f'{n_written}/{n_expected}':>14}"
              f"{pct:>6.1f}%{size/2**30:>10.2f}G{spr:>10.4f}{_fmt_hms(secs):>10}")

    print("  " + "-" * 92)
    pct = 100.0 * tot_done / tot_expected if tot_expected else 0.0
    print(f"  {'':<3}{'RAZEM':<18}{'':<3}{'':<12}{f'{tot_done}/{tot_expected}':>14}"
          f"{pct:>6.1f}%{tot_bytes/2**30:>10.2f}G"
          f"{(float(np.mean(rates)) if rates else 0.0):>10.4f}{_fmt_hms(tot_seconds):>10}")
    if rates and tot_expected > tot_done:
        rate = float(np.mean(rates))
        remaining_renders = (tot_expected - tot_done) * MEAN_N_SPEC
        print(f"\n  pozostalo ~{remaining_renders:,.0f} renderow".replace(",", " ") +
              f" x {rate:.4f} s = {_fmt_hms(remaining_renders * rate)} "
              f"({remaining_renders * rate / 3600:.1f} h)")
    print(f"  H = scena held-out ({', '.join(HELD_OUT)})")
    print("=" * 96)
    return 0


