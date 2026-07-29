"""Obsluga przerwania, logowanie i drobne formatery czasu."""

import signal
import sys
import time
from pathlib import Path

from .paths import scene_log

# Przerwanie — SIGINT/SIGTERM konczy BIEZACA probke, zapisuje ja i zamyka plik.
# Przerwanie w srodku N-powtorzen jednej probki jest zabronione (GENERATOR_PARAMS
# §4: checkpoint na granicy PROBKI jest bezpieczny, e1_checkpoint_boundary_merge,
# Wilcoxon p=0.949 — ale granica probki, nie granica renderu).
# ---------------------------------------------------------------------------
_INTERRUPTED = False


def _install_signal_handlers(log):
    def handler(signum, _frame):
        global _INTERRUPTED
        name = signal.Signals(signum).name
        if _INTERRUPTED:
            log.warning("%s po raz drugi — przerywam natychmiast (plik moze byc niekompletny)", name)
            raise KeyboardInterrupt(name)
        _INTERRUPTED = True
        log.warning("%s — koncze biezaca probke, zapisuje i zamykam plik czysto. "
                    "Wznowienie: --scene <scena> --resume", name)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# Logowanie
# ---------------------------------------------------------------------------
def setup_logging(scene=None, log_path=None):
    import logging

    log = logging.getLogger("gen")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    path = log_path if log_path is not None else (scene_log(scene) if scene else None)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log



def _fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"



def interrupted():
    """Czy przyszedl juz sygnal przerwania.

    Akcesor, a nie import samej zmiennej: `from .runtime import _INTERRUPTED`
    zwiazalby w module wolajacym KOPIE wartosci z chwili importu, wiec pozniejsze
    ustawienie flagi przez handler nigdy by tam nie dotarlo.
    """
    return _INTERRUPTED
