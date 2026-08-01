#!/usr/bin/env python3
"""Pulpit sterujacy generacja datasetu ech (Visual Echoes 2.0).

Nakladka na `generate_echo_dataset.py` — nie duplikuje jego logiki ani sciezek,
tylko je importuje, wiec zmiana struktury repo dalej wymaga edycji w jednym
miejscu (bloku sciezek w generatorze).

    python my-operations/echo_ctl.py              # pulpit interaktywny
    python my-operations/echo_ctl.py status       # tabelka i wyjscie
    python my-operations/echo_ctl.py watch        # podglad na zywo
    python my-operations/echo_ctl.py next         # uruchom kolejna scene wg §4.2
    python my-operations/echo_ctl.py start <scena>
    python my-operations/echo_ctl.py regen <scena>   # od zera, biezacymi parametrami
    python my-operations/echo_ctl.py stop
    python my-operations/echo_ctl.py verify <scena>

KOLEJKA — generacja bez nadzoru (np. przez noc):

    python my-operations/echo_ctl.py queue                  # POKAZ kolejke (nie zmienia jej)
    python my-operations/echo_ctl.py queue all              # wszystko, co jeszcze trzeba
    python my-operations/echo_ctl.py queue room_0 office_0  # ustaw dokladnie te sceny
    python my-operations/echo_ctl.py queue add hotel_0
    python my-operations/echo_ctl.py queue rm apartment_0   # dziala NA ZYWO
    python my-operations/echo_ctl.py queue clear            # oproznij (nie przerywa sceny)
    python my-operations/echo_ctl.py stop                   # zatrzymuje kolejke I scene

W pulpicie: [k] otwiera liste z przelacznikami — numery/nazwy przelaczaja scene,
`w` wszystkie, `c` zadna, `s` ZAPISUJE, Enter ANULUJE bez zmian.

EDYCJA DZIALA NA ZYWO. Nadzorca czyta plik kolejki przed kazda scena, wiec
dodawanie i usuwanie w trakcie generacji nie wymaga niczego zatrzymywac — zmiana
obowiazuje od nastepnej sceny. `queue clear` oprozni kolejke, ale NIE przerwie
trwajacej sceny; do przerwania calosci sluzy `stop`.

Zadna zmiana nie wchodzi w zycie bez jawnego [s] / podpolecenia. Golе `queue` tylko
wypisuje stan — wczesniej puste wejscie znaczylo "zakolejkuj wszystko", co pozwalalo
przypadkowym Enterem wrzucic 17 scen bez mozliwosci wycofania.

Kolejka to OSOBNY PROCES-NADZORCA: uruchamia generator na jedna scene, czeka na
jego zakonczenie i bierze nastepna. Nie jest to petla wewnatrz generatora, bo
jeden Simulator na proces to twarde zalozenie projektu (konstruowanie
wielu Simulatorow w jednym procesie potrafi zawiesic GPU). Nadzorca sam nigdy nie
tworzy Simulatora.

  * Stan trzymany w `.queue.json` w katalogu wariantu (zapis atomowy), wiec
    `status` w drugim terminalu widzi kolejke, a po zabiciu nadzorcy wiadomo,
    gdzie sie zatrzymala. Postep kazdej scenie i tak jest w jej pliku HDF5.
  * `stop` KASUJE plik kolejki, a nadzorca sprawdza go przed kazda scena — bez
    tego generator przerwany SIGINT-em wychodzi kodem 0 i kolejka wzielaby po
    prostu nastepna scene.
  * Scena juz rozpoczeta jest wznawiana (`--resume`), nie liczona od zera.
  * Po niepowodzeniu scena jest ponawiana RAZ (z `--resume`, wiec nic nie ginie),
    a po drugiej porazce cala kolejka jest PRZERYWANA. Typowa przyczyna to
    zawieszony GPU — bez tego kolejne sceny dopisywalyby bledy przez cala noc.
  * Kolejke mozna zalozyc, GDY GENERACJA JUZ DZIALA — nadzorca poczeka na wolne
    GPU i ruszy sam po zakonczeniu biezacej sceny. Nie trzeba przy tym siedziec.
  * Sceny juz kompletne nadzorca pomija bez konstruowania Simulatora, wiec mozna
    bezpiecznie dokolejkowac takze te, ktora wlasnie sie konczy.

DWA WARIANTY DATASETU. Kazde polecenie przyjmuje `--variant` (albo `-v`), ktory
mozna podac w dowolnym miejscu wiersza:

    python my-operations/echo_ctl.py next                        # wariant glowny
    python my-operations/echo_ctl.py --variant patched next       # wariant dodatkowy
    python my-operations/echo_ctl.py -v patched start frl_apartment_2
    python my-operations/echo_ctl.py -v patched queue             # cala kolejka wariantu

W pulpicie interaktywnym wariant przelacza sie klawiszem [w] — bez wychodzenia
i restartu. Przelaczenie zmienia tylko WIDOK i to, co uruchomia kolejne polecenia;
na trwajaca generacje nie ma zadnego wplywu (ma wlasny, zamrozony w chwili startu
zestaw sciezek). Po przelaczeniu naglowek ostrzega, jesli w drugim wariancie cos
dziala — GPU jest jedno. `start` w takiej sytuacji odmowi, a `queue` po prostu
poczeka: kolejke mozna zalozyc w trakcie generacji i wystartuje sama po niej.

  * `main` (domyslny) — geometria ORYGINALNA Repliki, 18 scen, 62 640 probek.
    Tylko ten wariant zachowuje zgodnosc RGB/depth z VisualEchoes, wiec tylko on
    jest porownywalny z praca zrodlowa.
  * `patched` — sceny z domknietymi dziurami (measurements/patch_scene_holes.py),
    **10 scen, 44 064 probki**. Tylko te 10 mialo dziure; pozostale 8 jest
    szczelnych, ich siatka jest w obu wariantach identyczna, wiec generowanie ich
    po raz drugi byloby strata czasu GPU — do treningu wariant dodatkowy sklada
    sie z tych 10 scen PLUS 8 scen szczelnych z wariantu glownego.

Warianty maja rozdzielne katalogi wyjsciowe (`outputs/echoes_36deg` i
`outputs/echoes_36deg_patched`), wlasne cache indeksu i wlasne statusy, wiec nie
da sie ich pomylic ani nadpisac. Kazdy plik HDF5 ma atrybut `variant`, a `scene_id`
wskazuje faktycznie uzyta siatke. Pulpit ostrzega, jesli generacja dziala
w INNYM wariancie niz ogladany — GPU jest jedno.

Uzasadnienie fizyczne wariantu dodatkowego i jego ograniczenia:
RAPORT_SESJI_2026-07-26_29.md §2.13-§2.15 (ucieczka promieni 22 % -> 0.00 %,
zgodnosc z Eyringiem 0.41x -> 1.00x, ale sceny zalatane sa mierzalnie BARDZIEJ
wyidealizowane niz nienaruszone skany, p = 0.0032).

ODPORNOSC NA ZERWANIE SESJI. Generacja jest uruchamiana przez `setsid`, wiec
proces trafia do wlasnej sesji i wlasnej grupy procesow: zamkniecie terminala
albo zerwanie SSH wysyla SIGHUP tylko do grupy terminala, ktorej generator juz
nie jest czlonkiem. Dlatego zamkniecie tego pulpitu NIGDY nie zatrzymuje
generacji — do zatrzymania sluzy wylacznie polecenie `stop` (SIGINT, ktory
generator obsluguje konczac biezaca probke).

Po ponownym uruchomieniu pulpit sam wykrywa dzialajaca generacje, skanujac
/proc po wierszu polecen — nie polega na pliku PID, ktory po twardym zabiciu
procesu zostawalby nieaktualny.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_echo_dataset as G  # noqa: E402  — jedyne zrodlo sciezek i stalych
from echo_core import paths as P  # noqa: E402

# DLACZEGO OSOBNY IMPORT `paths`, skoro G juz wszystko re-eksportuje: `OUT_ROOT` jest
# STALA, a `from ... import OUT_ROOT` w generatorze zwiazalo jej kopie w chwili importu.
# set_variant() podmienia globalna w echo_core.paths, wiec G.OUT_ROOT po zmianie
# wariantu jest NIEAKTUALNE — funkcje sciezek (G.scene_h5 itp.) czytaja globalna
# w czasie wywolania i dzialaja, ale sama stala nie. Zlapane pomiarowo: kolejka
# ladowala plik stanu w katalogu wariantu glownego mimo --variant patched.

GEN_SCRIPT = G.SCRIPT_PATH

# Wariant datasetu. Ustawiany raz w main() z argumentu --variant / wariant=...,
# potem czytany przez SCENES() i sciezki w G. Kazdy wariant ma wlasny katalog
# wyjsciowy, wiec statusy i cache sie nie mieszaja.
VARIANT = "main"


def SCENES():
    """Sceny biezacego wariantu — `patched` ma tylko te, ktore maja late."""
    return G.scenes_for_variant()


def scene_index_cache():
    # sciezka MUSI byc liczona po ustawieniu wariantu, dlatego funkcja, nie stala
    return P.OUT_ROOT / ".scene_index.json"

# --- kolory: wylaczane automatycznie, gdy wyjscie nie jest terminalem --------
_TTY = sys.stdout.isatty()


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if _TTY else text


DIM = lambda s: c(s, "2")
BOLD = lambda s: c(s, "1")
GREEN = lambda s: c(s, "32")
YELLOW = lambda s: c(s, "33")
RED = lambda s: c(s, "31")
CYAN = lambda s: c(s, "36")


# ---------------------------------------------------------------------------
# Wykrywanie dzialajacej generacji
# ---------------------------------------------------------------------------
def find_running():
    """-> lista (pid, scena) faktycznie dzialajacych generacji.

    Skanujemy /proc zamiast uzywac `pgrep -f`, bo ten ostatni dopasowuje
    dowolny fragment wiersza polecen i lapie takze opakowania powloki
    (`bash -c "... generate_echo_dataset.py ..."`), a nawet samego siebie.
    Tutaj wymagamy, zeby argv[0] byl interpreterem pythona, a nazwa skryptu
    wystapila jako osobny argument.
    """
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if len(argv) < 2 or "python" not in Path(argv[0]).name:
            continue
        if not any(Path(a).name == GEN_SCRIPT.name for a in argv[1:]):
            continue
        scene, variant = None, "main"
        for i, a in enumerate(argv):
            if a == "--scene" and i + 1 < len(argv):
                scene = argv[i + 1]
            elif a == "--variant" and i + 1 < len(argv):
                variant = argv[i + 1]
        out.append((int(entry.name), scene, variant))
    return out


def scene_index():
    """scena -> liczba lokalizacji. Cache na dysku, bo liczenie wymaga
    wczytania 913 MB pkl (0.9 s) — za duzo jak na widok odswiezany co 2 s."""
    cache = scene_index_cache()
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    idx = {}
    for s in SCENES():
        try:
            idx[s] = len(G.load_scene_locations(s)[0])
        except Exception:
            idx[s] = 0
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx, indent=2))
    return idx


# ---------------------------------------------------------------------------
# Stan sceny
# ---------------------------------------------------------------------------
def scene_state(scene, index, running_map):
    path = G.scene_h5(scene)
    st = {"scene": scene, "written": 0, "expected": index.get(scene, 0) * G.N_ANGLES,
          "size": 0, "spr": 0.0, "pid": running_map.get(scene), "state": "brak",
          "held_out": scene in G.HELD_OUT}
    if not path.exists():
        if st["pid"]:
            st["state"] = "start..."
        return st

    st["size"] = path.stat().st_size
    try:
        f = G._open_readonly(path)
        try:
            w = f["written"][:]
            st["written"] = int(w.sum())
            st["expected"] = int(f.attrs.get("n_samples_expected", w.size))
            st["spr"] = float(f.attrs.get("seconds_per_render", 0.0))
            st["seconds"] = sum(r.get("wall_seconds", 0.0)
                                for r in json.loads(f.attrs.get("runs", "[]")))
            # Rendery/probke: mierzone N, ktore w tej scenie faktycznie wyszlo
            # z reguly adaptacyjnej. Uzywane zamiast zalozonego 9.83 do ETA,
            # bo rozklad szumu okazal sie mocno zalezec od sceny.
            #
            # Zrodlem jest zbior `n_total` (zapisywany przy KAZDEJ probce), a nie
            # atrybut `renders_total` — ten drugi aktualizuje sie dopiero przy
            # zamknieciu pliku, wiec w trakcie przebiegu dzielilby stara liczbe
            # renderow przez biezaca liczbe probek i zanizal wynik.
            if st["written"]:
                nt = f["n_total"][:][w.astype(bool)]
                if nt.size:
                    st["n_per_sample"] = float(nt.mean())
        finally:
            f.close()
    except Exception:
        sp = G.scene_progress(scene)
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                st["written"] = int(d.get("n_written", 0))
                st["expected"] = int(d.get("n_samples_expected", st["expected"]))
                st["spr"] = float(d.get("seconds_per_render") or 0.0)
            except (OSError, ValueError):
                pass

    if st["expected"] and st["written"] >= st["expected"]:
        st["state"] = "gotowa"
    elif st["pid"]:
        st["state"] = "W TOKU"
    elif st["written"]:
        st["state"] = "przerwana"
    else:
        st["state"] = "pusta"
    return st


def all_states():
    index = scene_index()
    running_map = {s: p for p, s, v in find_running() if s and v == VARIANT}
    return [scene_state(s, index, running_map) for s in SCENES()], running_map


def tail_log(scene, n=3):
    p = G.scene_log(scene)
    if not p.exists():
        return []
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return []
    keep = [l for l in lines if " lok " in l or "PRZERWAN" in l or "BLAD" in l
            or "Simulator zbudowany" in l or "decyzja odtworzona" in l]
    return (keep or lines)[-n:]


def parse_progress(scene):
    """Ostatni wiersz postepu -> (numer, ile, N, sr_N, ETA) albo None."""
    for line in reversed(tail_log(scene, n=12)):
        m = re.search(r"lok\s+(\d+)/(\d+)\s+id=(\d+).*?N=(\d+).*?sr\.N\s+([\d.]+).*?ETA\s+(\S+)", line)
        if m:
            return dict(i=int(m.group(1)), n=int(m.group(2)), loc=int(m.group(3)),
                        N=int(m.group(4)), meanN=float(m.group(5)), eta=m.group(6))
    return None


# ---------------------------------------------------------------------------
# Prezentacja
# ---------------------------------------------------------------------------
def bar(frac, width=24):
    frac = max(0.0, min(1.0, frac))
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full)


def fmt_size(b):
    return f"{b/2**30:.2f} GB" if b >= 2**30 else f"{b/2**20:.0f} MB"


def header(states, running_map):
    done = sum(1 for s in states if s["state"] == "gotowa")
    written = sum(s["written"] for s in states)
    expected = sum(s["expected"] for s in states)
    size = sum(s["size"] for s in states)
    rates = [s["spr"] for s in states if s["spr"] > 0]
    rate = sum(rates) / len(rates) if rates else G.S_PER_RENDER_SPEC

    width = min(shutil.get_terminal_size((80, 24)).columns, 100)
    tag = ("wariant GLOWNY (geometria oryginalna)" if VARIANT == "main"
           else f"wariant DODATKOWY '{VARIANT}' (sceny z domknietymi dziurami)")
    print(BOLD("  Visual Echoes 2.0 · generator ech 36-orientacyjnych") + DIM(f"  ·  {tag}"))
    foreign = [(p, sc, v) for p, sc, v in find_running() if sc and v != VARIANT]
    if foreign:
        for pid, sc, v in foreign:
            print(YELLOW(f"  UWAGA: w innym wariancie ({v}) dziala {sc} (pid {pid}) — "
                         f"jeden Simulator na GPU!"))
    print(DIM("  " + "─" * (width - 2)))

    q = read_queue()
    if q:
        alive = supervisor_alive(q)
        # UWAGA: wlasne nazwy z przedrostkiem q_ — `done` i `left` sa juz uzywane
        # wyzej w tej funkcji jako LICZNIKI scen, a przyslonienie ich lista dawalo
        # w naglowku "sceny ['office_2']/10".
        q_left, q_done = q.get("pending") or [], q.get("done") or []
        est = sum(next((x["expected"] for x in states if x["scene"] == sc), 0) for sc in q_left)
        eta = f"{est * rate * G.MEAN_N_SPEC / 3600:.0f} h" if est else "—"
        tag = GREEN("KOLEJKA") if alive else RED("KOLEJKA (nadzorca nie zyje)")
        # Scena uruchomiona RECZNIE nie jest czlonkiem kolejki, ale ma byc widoczna
        # w tej samej linii — inaczej obraz jest rozjechany: kolejka mowi "czekam",
        # a nie widac na co. Dopisujemy ja z adnotacja, skad sie wziela.
        cur = q.get("current") or (next(iter(running_map), None) if running_map else None)
        cur_note = ""
        if cur:
            cur_note = (f" · w toku: {BOLD(cur)}"
                        + ("" if cur == q.get("current") else DIM(" (uruchomiona recznie)")))
        print(f"  {tag}{cur_note} · gotowe {len(q_done)} · pozostalo {len(q_left)} "
              + DIM(f"({', '.join(q_left[:4])}{'…' if len(q_left) > 4 else ''}) ~{eta}"))
        if q.get("failed"):
            print(RED(f"  kolejka przerwana na: {', '.join(q['failed'])} — patrz {queue_log().name}"))

    active = [s for s in states if s["state"] in ("W TOKU", "start...")]
    if active:
        for s in active:
            pr = parse_progress(s["scene"])
            frac = s["written"] / s["expected"] if s["expected"] else 0
            line = (f"  {GREEN('▶ W TOKU')}  {BOLD(s['scene']):<26} {bar(frac)} "
                    f"{100*frac:5.1f}%  {s['written']}/{s['expected']}")
            print(line)
            extra = f"pid {s['pid']}"
            if pr:
                extra = (f"lok {pr['i']}/{pr['n']} (id={pr['loc']})  N={pr['N']}  "
                         f"śr.N {pr['meanN']:.1f}  ETA {pr['eta']}  " + extra)
            print(DIM(f"            {extra}"))
    else:
        stalled = [s for s in states if s["state"] == "przerwana"]
        if stalled:
            names = ", ".join(s["scene"] for s in stalled)
            print(f"  {YELLOW('◼ nic nie działa')}   przerwane sceny: {names}")
            print(DIM(f"            wznowienie: opcja [3] albo `echo_ctl.py start {stalled[0]['scene']}`"))
        else:
            print(f"  {DIM('◼ nic nie działa')}")

    print(DIM("  " + "─" * (width - 2)))
    # Pozostaly czas liczymy PER SCENA: kazda scena ma wlasne N (mierzone, jesli
    # ma juz dane), a nieruszone dostaja srednia ze specyfikacji. Uśrednianie
    # jednego N po calym zbiorze byloby mylace — office_1 wyszla ~21, ale to
    # 0.9 % lokalizacji, wiec rozciagniecie jej na 18 scen zawyzalo ETA dwukrotnie.
    remaining_s = 0.0
    n_seen = []
    for s in states:
        left = max(s["expected"] - s["written"], 0)
        n_scene = s.get("n_per_sample") or G.MEAN_N_SPEC
        if s.get("n_per_sample"):
            n_seen.append(s["n_per_sample"])
        remaining_s += left * n_scene * rate
    n_note = (f"N zmierzone dla {len(n_seen)}/{len(states)} scen, reszta {G.MEAN_N_SPEC} wg spec"
              if n_seen else f"N {G.MEAN_N_SPEC} wg spec")
    print(f"  sceny {BOLD(f'{done}/{len(states)}')} · próbki {written}/{expected} "
          f"({100*written/max(expected,1):.1f}%) · {fmt_size(size)} · "
          f"pozostało ~{remaining_s/3600:.0f} h")
    print(DIM(f"        przy {rate:.3f} s/render · {n_note}"))
    return width


STATE_COLOUR = {"gotowa": GREEN, "W TOKU": GREEN, "start...": GREEN,
                "przerwana": YELLOW, "NIECZYTELNY": RED}


def table(states):
    width = min(shutil.get_terminal_size((80, 24)).columns, 100)
    print()
    print(DIM(f"  {'#':<3}{'scena':<21}{'stan':<11}{'próbki':>16}"
              f"{'%':>8}{'rozmiar':>10}{'s/rend':>9}"))
    for i, s in enumerate(states, 1):
        pct = 100 * s["written"] / s["expected"] if s["expected"] else 0
        # Kolorowanie dokladamy PO sformatowaniu do stalej szerokosci — kody ANSI
        # licza sie do dlugosci napisu, wiec kolumny inaczej by sie rozjechaly.
        state_cell = STATE_COLOUR.get(s["state"], DIM)(f"{s['state']:<11}")
        progress = f"{s['written']}/{s['expected']}"
        name = f"{s['scene']} {CYAN('H')}" if s["held_out"] else s["scene"]
        pad = 21 + (len(CYAN('H')) - 1 if s["held_out"] else 0)
        rate = f"{s['spr']:.4f}" if s["spr"] else "-"
        print(f"  {i:<3}{name:<{pad}}{state_cell}{progress:>16}"
              f"{pct:>7.1f}%{fmt_size(s['size']):>10}{rate:>9}")
    print(DIM(f"  H = scena held-out ({', '.join(G.HELD_OUT)})"))
    print(DIM("  " + "─" * (width - 2)))


# ---------------------------------------------------------------------------
# Akcje
# ---------------------------------------------------------------------------
def next_scene(states):
    for s in states:
        if s["state"] in ("pusta", "brak", "przerwana"):
            return s
    return None


# ---------------------------------------------------------------------------
# Kolejka scen — generacja bez nadzoru (np. na noc)
# ---------------------------------------------------------------------------
# Kolejka to OSOBNY PROCES-NADZORCA, ktory uruchamia generator na jedna scene,
# czeka na jego zakonczenie i bierze nastepna. Dlaczego nie petla w jednym
# procesie generatora: jeden Simulator na proces to twarde zalozenie projektu
# (konstruowanie wielu Simulatorow w jednym procesie potrafi zawiesic
# GPU), wiec kazda scena MUSI dostac swoj proces. Nadzorca sam nigdy nie tworzy
# Simulatora, wiec nie zuzywa budzetu konstrukcji.
#
# Stan trzymamy w pliku JSON w katalogu wariantu, nie w pamieci nadzorcy — dzieki
# temu `status` w drugim terminalu widzi kolejke, a po zabiciu nadzorcy wiadomo,
# gdzie sie zatrzymala. USUNIECIE tego pliku jest sygnalem STOP: nadzorca sprawdza
# go przed kazda scena i konczy prace, gdy zniknie (tak dziala `stop`).


def queue_file():
    return P.OUT_ROOT / ".queue.json"


def queue_log():
    return P.OUT_ROOT / "queue.log"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def read_queue():
    try:
        return json.loads(queue_file().read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_queue(q):
    p = queue_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    # zapis atomowy: `status` w drugim terminalu nigdy nie przeczyta polowy pliku
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(q, indent=2))
    tmp.replace(p)


def supervisor_alive(q):
    """Czy nadzorca z pliku kolejki nadal zyje.

    Ten sam rygor co w find_running(): sprawdzamy wiersz polecen procesu, zeby
    przypadkowe ponowne uzycie PID-u nie wygladalo jak dzialajaca kolejka.
    """
    pid = (q or {}).get("pid")
    if not pid:
        return False
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (OSError, PermissionError):
        return False
    argv = [a.decode(errors="replace") for a in argv if a]
    return "_run-queue" in argv and any(Path(a).name == Path(__file__).name for a in argv)


def scene_complete(scene):
    """Czy plik sceny ma juz wszystkie probki. Tanio, bez pkl i bez GPU."""
    path = G.scene_h5(scene)
    if not path.exists():
        return False
    try:
        f = G._open_readonly(path)
        try:
            w = f["written"][:]
            return int(w.sum()) >= int(f.attrs.get("n_samples_expected", w.size))
        finally:
            f.close()
    except Exception:
        return False


def queue_start(scenes, states=None):
    """Zaklada kolejke i odpala nadzorce w tle."""
    # Trwajaca generacja NIE blokuje zalozenia kolejki. Nadzorca ma wlasna petle
    # czekania na wolne GPU, wiec normalnym scenariuszem jest "odpalilem jedna scene
    # recznie, teraz dokolejkuje reszte, zeby poszla po niej" — wczesniejsza odmowa
    # byla nadmiarowa i wymuszala siedzenie przy komputerze do konca sceny.
    running = find_running()
    old = read_queue()
    if old and supervisor_alive(old):
        print(RED(f"  Kolejka juz dziala (pid {old['pid']})."))
        print(DIM(f"  pozostalo: {', '.join(old.get('pending') or []) or '—'}"))
        return 1

    if states is None:
        states, _ = all_states()
    if not scenes:
        # domyslnie wszystko, co jeszcze czegos potrzebuje, w kolejnosci z §4.2
        scenes = [s["scene"] for s in states if s["state"] in ("pusta", "brak", "przerwana")]
    unknown = [s for s in scenes if s not in SCENES()]
    if unknown:
        print(RED(f"  Nieznane sceny w wariancie {VARIANT}: {', '.join(unknown)}"))
        return 1
    seen, uniq = set(), []
    for s in scenes:                       # bez duplikatow, kolejnosc zachowana
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if old and old.get("failed"):
        print(YELLOW(f"  poprzednia kolejka przerwala sie na: {', '.join(old['failed'])}"))
    return queue_apply(uniq, states, running)


def queue_apply(pending, states=None, running=None):
    """Ustawia zawartosc kolejki i pilnuje, zeby nadzorca dzialal.

    Sluzy i do zalozenia kolejki, i do EDYCJI dzialajacej: nadzorca czyta plik
    przed kazda scena, wiec zmiana `pending` dziala na zywo i nie wymaga
    zatrzymywania czegokolwiek. Pusta lista = kolejka wygasa po biezacej scenie
    (nadzorca sam wyjdzie), ale trwajaca generacja NIE jest przerywana — do tego
    sluzy `stop`.
    """
    if states is None:
        states, _ = all_states()
    if running is None:
        running = find_running()
    old = read_queue() or {}
    alive = supervisor_alive(old)

    write_queue({"variant": VARIANT, "pending": list(pending),
                 "done": old.get("done") or [], "failed": old.get("failed") or [],
                 "current": old.get("current"),
                 "started": old.get("started") or datetime.now().isoformat(timespec="seconds"),
                 "pid": old.get("pid") if alive else None})

    if not pending:
        print(YELLOW("  Kolejka oprozniona." + (" Nadzorca zakonczy sie po biezacej scenie."
                                               if alive else "")))
        print(DIM("  Trwajaca generacja NIE zostala przerwana — do tego sluzy `stop`."))
        return 0

    total = sum(next((x["expected"] for x in states if x["scene"] == sc), 0) for sc in pending)
    if alive:
        print(GREEN(f"  Kolejka zaktualizowana na zywo (nadzorca pid {old['pid']}): "
                    f"{len(pending)} scen, {total} probek"))
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), "_run-queue",
               "--variant", VARIANT]
        # start_new_session: nadzorca przezywa zamkniecie terminala, dokladnie tak samo
        # jak pojedyncza generacja uruchamiana przez start()
        with open(queue_log(), "ab") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    cwd=str(G.REPO_ROOT))
        print(GREEN(f"  Kolejka uruchomiona (pid {proc.pid}): {len(pending)} scen, "
                    f"{total} probek"))
    for i, sc in enumerate(pending, 1):
        print(DIM(f"    {i}. {sc}"))
    if running:
        pid, sc, v = running[0]
        print(YELLOW(f"  Trwa generacja {sc} [{v}] (pid {pid}) — kolejka poczeka na wolne GPU"))
        print(DIM("  i wystartuje sama, gdy ta scena sie skonczy. Nie trzeba nic klikac."))
    print(DIM(f"  log kolejki: {queue_log()}"))
    print(DIM("  Zamkniecie pulpitu ani zerwanie SSH NIE zatrzyma kolejki."))
    print(DIM("  Zatrzymanie calosci: `stop`.  Sama kolejka: [k] -> c, albo `queue clear`."))
    time.sleep(1.5)
    return 0


def queue_candidates(states):
    """Sceny, ktore moga trafic do kolejki: niegotowe + te juz w niej siedzace."""
    q = read_queue() or {}
    need = {x["scene"] for x in states if x["state"] in ("pusta", "brak", "przerwana")}
    need |= set(q.get("pending") or [])
    return [sc for sc in SCENES() if sc in need]


def queue_edit(states):
    """Interaktywny edytor kolejki: lista z zaznaczeniem, przelaczanie numerami.

    DLACZEGO Enter = ANULUJ, a nie "zakolejkuj wszystko": poprzednia wersja pytala
    o liste i puste wejscie traktowala jako "wszystko", wiec przypadkowy Enter
    wrzucal do kolejki 17 scen bez mozliwosci wycofania. Tutaj zadna zmiana nie
    wchodzi w zycie bez jawnego [s].
    """
    cand = queue_candidates(states)
    if not cand:
        print(GREEN("  Nie ma czego kolejkowac — wszystkie sceny wariantu gotowe."))
        return 0
    q = read_queue() or {}
    sel = [sc for sc in cand if sc in (q.get("pending") or [])]
    by_scene = {x["scene"]: x for x in states}
    orig = list(sel)

    while True:
        print()
        print(BOLD(f"  KOLEJKA — wariant {VARIANT}") +
              DIM("   [x] = w kolejce, kolejnosc jak w §4.2"))
        for i, sc in enumerate(cand, 1):
            mark = GREEN("[x]") if sc in sel else DIM("[ ]")
            st = by_scene.get(sc, {})
            lok = st.get("expected", 0) // G.N_ANGLES
            note = ""
            if st.get("state") == "przerwana":
                note = YELLOW(" wznowi")
            elif st.get("held_out"):
                note = DIM(" held-out")
            print(f"   {mark} {i:>2}. {sc:<18}{lok:>4} lok.{note}")
        est = sum(by_scene.get(sc, {}).get("expected", 0) for sc in sel)
        print(DIM(f"   wybrane: {len(sel)} scen, {est} probek"))
        print(DIM("   numery/nazwy = przelacz  ·  w = wszystkie  ·  c = zadna  ·  "
                  "s = ZAPISZ  ·  Enter = anuluj"))
        try:
            raw = input("  kolejka> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if raw == "":
            print(DIM("  anulowane — kolejka bez zmian"))
            return 0
        if raw == "s":
            if sel == orig:
                print(DIM("  bez zmian"))
                return 0
            return queue_apply(sel, states)
        if raw == "w":
            sel = list(cand)
            continue
        if raw == "c":
            sel = []
            continue
        for tok in raw.replace(",", " ").split():
            target = None
            if tok.isdigit() and 1 <= int(tok) <= len(cand):
                target = cand[int(tok) - 1]
            elif tok in cand:
                target = tok
            if target is None:
                print(RED(f"  nie rozpoznaje: {tok}"))
                continue
            if target in sel:
                sel.remove(target)
            else:
                # wstawiamy tak, zeby zachowac kolejnosc SCENE_ORDER
                sel = [sc for sc in cand if sc in set(sel) | {target}]


def run_queue():
    """Nadzorca. Uruchamiany tylko przez queue_start(), nie recznie."""
    q = read_queue()
    if not q:
        print(f"[{_ts()}] brak pliku kolejki — koniec")
        return 1
    q["pid"] = os.getpid()
    write_queue(q)
    print(f"[{_ts()}] KOLEJKA start, wariant {q['variant']}, sceny: {', '.join(q['pending'])}")

    while True:
        q = read_queue()
        if q is None:
            print(f"[{_ts()}] plik kolejki usuniety — zatrzymuje kolejke (to jest `stop`)")
            return 0
        if not q.get("pending"):
            break
        scene = q["pending"][0]

        # ktos mogl w miedzyczasie odpalic generacje recznie — GPU jest jedno
        other = [r for r in find_running()]
        if other:
            print(f"[{_ts()}] czekam: GPU zajete przez {other[0][1]} (pid {other[0][0]})")
            time.sleep(30)
            continue

        # Scena moze byc juz gotowa — np. dokolejkowano recznie te, ktora wlasnie
        # sie konczyla. Pomijamy bez konstruowania Simulatora.
        if scene_complete(scene):
            print(f"[{_ts()}] {scene} jest juz kompletna — pomijam")
            q["pending"].remove(scene)
            q.setdefault("done", []).append(scene)
            write_queue(q)
            continue

        q["current"] = scene
        write_queue(q)
        rc = 1
        for attempt in (1, 2):
            partial = G.scene_h5(scene).exists()
            cmd = [sys.executable, str(GEN_SCRIPT), "--scene", scene,
                   "--variant", q["variant"]]
            if partial:
                cmd.append("--resume")
            G.scene_dir(scene).mkdir(parents=True, exist_ok=True)
            print(f"[{_ts()}] start {scene} (proba {attempt}"
                  f"{', --resume' if partial else ''})")
            with open(G.scene_stdout(scene), "ab") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=str(G.REPO_ROOT))
            if rc == 0:
                print(f"[{_ts()}] {scene} GOTOWA")
                break
            print(f"[{_ts()}] {scene} zakonczyla sie kodem {rc}")
            # Jedno ponowienie, bo --resume nie traci ani jednej probki, a awarie
            # bywaja przejsciowe. Po drugiej porazce PRZERYWAMY cala kolejke:
            # typowa przyczyna to zawieszony GPU, a wtedy kolejne
            # sceny tylko dopisywalyby bledy przez cala noc.
            if attempt == 1:
                print(f"[{_ts()}] ponawiam raz z --resume (nic nie ginie)")
                time.sleep(15)

        q = read_queue()
        if q is None:
            print(f"[{_ts()}] plik kolejki usuniety w trakcie — koniec")
            return 0
        if scene in q.get("pending", []):
            q["pending"].remove(scene)
        q["current"] = None
        if rc == 0:
            q.setdefault("done", []).append(scene)
        else:
            q.setdefault("failed", []).append(scene)
            q["pending"] = []
            print(f"[{_ts()}] PRZERWANIE kolejki po dwoch nieudanych probach na {scene}")
        write_queue(q)

    q = read_queue() or q
    print(f"[{_ts()}] KOLEJKA koniec — gotowe: {', '.join(q.get('done') or []) or '—'}"
          f"{'; nieudane: ' + ', '.join(q['failed']) if q.get('failed') else ''}")
    return 0 if not q.get("failed") else 1


def scene_params(scene):
    """Parametry, ktorymi wygenerowano istniejacy plik sceny. -> dict albo None.

    Sluza porownaniu z parametrami BIEZACYMI: po zmianie N_MAX albo
    WARMUP_DISCARD stare sceny przestaja byc jednorodne z nowymi, a plik HDF5
    zapisuje uzyte wartosci w atrybutach — wiec da sie to wykryc, a nie zgadywac.
    """
    path = G.scene_h5(scene)
    if not path.exists():
        return None
    try:
        f = G._open_readonly(path)
    except Exception:
        return None
    try:
        return {k: (f.attrs[k].item() if hasattr(f.attrs[k], "item") else f.attrs[k])
                for k in ("n_max", "n_min", "n_probe", "warmup_discard",
                          "audio_sims_per_render", "signal_10deg", "target_snr")
                if k in f.attrs}
    finally:
        f.close()


CURRENT_PARAMS = {"n_max": G.N_MAX, "n_min": G.N_MIN, "n_probe": G.N_PROBE,
                  "warmup_discard": G.WARMUP_DISCARD, "audio_sims_per_render": 1,
                  "signal_10deg": G.SIGNAL_10DEG, "target_snr": G.TARGET_SNR}


def stale_params(scene):
    """-> lista (parametr, w pliku, teraz) dla roznic. Pusta = plik aktualny."""
    old = scene_params(scene)
    if old is None:
        return []
    out = []
    for k, now in CURRENT_PARAMS.items():
        if k in old and old[k] != now:
            out.append((k, old[k], now))
    return out


def regenerate(scene):
    """Generuje scene OD ZERA biezacymi parametrami (--force). NISZCZY stary plik."""
    running = find_running()
    if running:
        pid, sc, v = running[0]
        print(RED(f"  Generacja juz dziala: {sc} [wariant {v}] (pid {pid})."))
        return 1
    if scene not in SCENES():
        print(RED(f"  Nieznana scena w wariancie {VARIANT}: {scene}"))
        if VARIANT != "main" and scene in G.SCENE_ORDER:
            print(DIM("  Ta scena jest szczelna — nie ma laty, wiec w tym wariancie jej"))
            print(DIM("  geometria bylaby identyczna jak w 'main'. Uzyj wariantu glownego."))
        return 1

    path = G.scene_h5(scene)
    diffs = stale_params(scene)
    if path.exists():
        st = scene_state(scene, scene_index(), {})
        print(f"  Istniejacy plik: {st['written']}/{st['expected']} probek, "
              f"{fmt_size(st['size'])}")
        if diffs:
            print(YELLOW("  Parametry w pliku ROZNIA SIE od biezacych:"))
            for k, was, now in diffs:
                print(f"    {k:<24} plik: {was!s:<12} teraz: {now}")
        else:
            print(DIM("  Parametry w pliku sa zgodne z biezacymi — regeneracja nic nie zmieni."))
        print(RED(f"\n  --force NADPISZE ten plik od zera. Starego nie da sie odzyskac."))
        ans = input("  Wpisz nazwe sceny, zeby potwierdzic: ").strip()
        if ans != scene:
            print(DIM("  Anulowane."))
            return 1
    else:
        print(DIM(f"  {scene}: pliku nie ma, to bedzie zwykla generacja od zera."))

    G.scene_dir(scene).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(GEN_SCRIPT), "--scene", scene, "--force"]
    with open(G.scene_stdout(scene), "ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=str(G.REPO_ROOT))
    print(GREEN(f"  Regeneracja {scene} od zera, pid {proc.pid}"))
    print(DIM(f"  log: {G.scene_log(scene)}"))
    time.sleep(2)
    return 0


def start(scene, states=None):
    running = find_running()
    if running:
        pid, sc, v = running[0]
        print(RED(f"  Generacja juz dziala: {sc} [wariant {v}] (pid {pid})."))
        print(DIM("  Jeden Simulator na raz — rownolegle sceny biłyby sie o GPU."))
        return 1
    if scene not in SCENES():
        print(RED(f"  Nieznana scena w wariancie {VARIANT}: {scene}"))
        if VARIANT != "main" and scene in G.SCENE_ORDER:
            print(DIM("  Ta scena jest szczelna — nie ma laty, wiec w tym wariancie jej"))
            print(DIM("  geometria bylaby identyczna jak w 'main'. Uzyj wariantu glownego."))
        return 1

    partial = G.scene_h5(scene).exists()
    if partial:
        diffs = stale_params(scene)
        if diffs:
            print(YELLOW(f"  UWAGA: {scene} zaczeto innymi parametrami niz biezace:"))
            for k, was, now in diffs:
                print(f"    {k:<24} plik: {was!s:<12} teraz: {now}")
            print(DIM("  --resume dopisze probki NOWYMI parametrami -> scena bedzie mieszana."))
            print(DIM("  Dla jednorodnosci uzyj regeneracji: echo_ctl.py regen " + scene))
    cmd = [sys.executable, str(GEN_SCRIPT), "--scene", scene, "--variant", VARIANT]
    if partial:
        cmd.append("--resume")

    # Katalog sceny tworzymy TU, przed startem procesu — dzieki temu --status
    # w drugim terminalu widzi scene od pierwszej sekundy, a nie dopiero po
    # pierwszym flushu HDF5.
    G.scene_dir(scene).mkdir(parents=True, exist_ok=True)
    stdout_path = G.scene_stdout(scene)
    # setsid odcina proces od sesji terminala: zerwanie SSH wysyla SIGHUP do
    # grupy procesow terminala, do ktorej generator po tym wywolaniu nie nalezy.
    with open(stdout_path, "ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=str(G.REPO_ROOT))
    print(GREEN(f"  Uruchomiono {scene}" + (" (--resume)" if partial else "") +
                f", pid {proc.pid}"))
    print(DIM(f"  log: {G.scene_log(scene)}"))
    print(DIM("  Zamkniecie tego pulpitu NIE zatrzyma generacji."))
    time.sleep(2)
    return 0


def stop(scene=None):
    # Kolejke kasujemy PRZED wyslaniem SIGINT. Generator po przerwaniu konczy
    # biezaca probke i wychodzi kodem 0, wiec nadzorca bez tego wzialby po prostu
    # nastepna scene — a `stop` ma zatrzymac calosc. Brak pliku kolejki jest dla
    # nadzorcy sygnalem zakonczenia pracy.
    q = read_queue()
    if q:
        queue_file().unlink(missing_ok=True)
        left = q.get("pending") or []
        print(YELLOW(f"  kolejka zatrzymana; nieuruchomione sceny: "
                     f"{', '.join(left) if left else '—'}"))
    running = find_running()
    if not running:
        print(DIM("  Nic nie dziala."))
        return 0
    for pid, sc, v in running:
        if scene and sc != scene:
            continue
        try:
            os.kill(pid, signal.SIGINT)
            print(YELLOW(f"  SIGINT -> {sc} [{v}] (pid {pid})"))
            print(DIM("  Generator dokonczy biezaca probke, zapisze i zamknie plik. "
                      "Moze to potrwac do kilkunastu sekund."))
        except ProcessLookupError:
            print(DIM(f"  pid {pid} juz nie istnieje"))
    return 0


def watch(scene=None, interval=2.0):
    print(DIM("\n  Podglad na zywo. Ctrl+C konczy PODGLAD, nie generacje.\n"))
    time.sleep(0.8)
    lines_drawn = 0
    try:
        while True:
            states, running_map = all_states()
            active = [s for s in states if s["state"] in ("W TOKU", "start...")]
            target = scene or (active[0]["scene"] if active else None)
            if target is None:
                # Nic nie dziala — pokazujemy scene ostatnio ruszana zamiast
                # pustego ekranu: najczestszy moment na ten widok to chwila TUZ
                # PO zakonczeniu sceny, kiedy wynik jest wlasnie tym, co chcemy zobaczyc.
                touched = [s for s in states if s["written"]]
                if touched:
                    target = max(touched, key=lambda s: G.scene_h5(s["scene"]).stat().st_mtime)["scene"]

            buf = []
            if target is None:
                buf.append("  " + DIM("nic nie dziala, zaden plik nie istnieje — [n] zaczyna kolejna scene"))
            else:
                s = next(x for x in states if x["scene"] == target)
                pr = parse_progress(target)
                frac = s["written"] / s["expected"] if s["expected"] else 0
                buf.append(f"  {BOLD(target)}  {bar(frac, 30)} {100*frac:5.1f}%   "
                           f"{s['written']}/{s['expected']} próbek")
                if pr:
                    buf.append(f"  lokalizacja {pr['i']}/{pr['n']} (id={pr['loc']})   "
                               f"N={pr['N']}   śr.N {pr['meanN']:.1f}   "
                               f"{s['spr']:.4f} s/render   ETA {BOLD(pr['eta'])}")
                else:
                    buf.append(DIM("  (czekam na pierwsza ukonczona lokalizacje)"))
                buf.append(DIM(f"  stan: {s['state']}   pid: {s['pid'] or '-'}   "
                               f"plik: {fmt_size(s['size'])}"))
                buf.append("")
                for line in tail_log(target, 3):
                    buf.append(DIM("  " + line[:shutil.get_terminal_size((100, 24)).columns - 4]))

            if _TTY and lines_drawn:
                sys.stdout.write(f"\033[{lines_drawn}A\033[J")
            out = "\n".join(buf)
            print(out)
            lines_drawn = out.count("\n") + 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print(DIM("\n  (podglad zakonczony; generacja dziala dalej)\n"))
    return 0


def run_gen(*extra):
    return subprocess.call([sys.executable, str(GEN_SCRIPT), *extra,
                           "--variant", VARIANT], cwd=str(G.REPO_ROOT))


def pick_scene(states, prompt="  scena (numer albo nazwa): "):
    table(states)
    raw = input(prompt).strip()
    if not raw:
        return None
    order = SCENES()
    if raw.isdigit() and 1 <= int(raw) <= len(order):
        return order[int(raw) - 1]
    if raw in order:
        return raw
    print(RED(f"  Nieznana scena: {raw}"))
    return None


# ---------------------------------------------------------------------------
# Pulpit
# ---------------------------------------------------------------------------
MENU = """
  {p} podgląd na żywo          {v} weryfikuj scenę
  {n} uruchom następną         {t} tabela wszystkich scen
  {k} KOLEJKA scen (na noc)    {w} przełącz wariant (main/patched)
  {s} uruchom wybraną          {d} próbny przebieg (dry-run)
  {x} zatrzymaj generację      {r} REGENERUJ scenę od zera
  {q} wyjście
"""


def dashboard():
    while True:
        if _TTY:
            os.system("clear")
        print()
        states, running_map = all_states()
        header(states, running_map)
        nxt = next_scene(states)
        if nxt and not running_map:
            action = "wznowi" if nxt["state"] == "przerwana" else "zacznie"
            print(DIM(f"  następna w kolejce: {BOLD(nxt['scene'])} "
                      f"({nxt['expected']//G.N_ANGLES} lok.) — [n] {action}"))
        stale = [s["scene"] for s in states if s["written"] and stale_params(s["scene"])]
        if stale:
            print(YELLOW(f"  sceny wygenerowane STARYMI parametrami: {', '.join(stale)}"
                         "  — [r] regeneruje"))
        print(MENU.format(p=BOLD("p"), n=BOLD("n"), s=BOLD("s"), x=BOLD("x"), v=BOLD("v"),
                          t=BOLD("t"), d=BOLD("d"), r=BOLD("r"), q=BOLD("q"), k=BOLD("k"),
                          w=BOLD("w")))
        try:
            ch = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if ch in ("q", "quit", "exit"):
            if running_map:
                print(DIM(f"\n  Generacja {list(running_map)[0]} dziala dalej w tle.\n"))
            return 0
        elif ch == "p":
            watch()
        elif ch == "n":
            if nxt:
                start(nxt["scene"])
            else:
                print(GREEN(f"  Wszystkie {len(SCENES())} scen gotowe."))
            input(DIM("  [Enter]"))
        elif ch == "s":
            sc = pick_scene(states)
            if sc:
                start(sc)
            input(DIM("  [Enter]"))
        elif ch == "w":
            other = "patched" if VARIANT == "main" else "main"
            if switch_variant(other):
                print(GREEN(f"  wariant: {other}"))
                # Jesli w starym wariancie cos dziala, naglowek nowego widoku sam
                # o tym ostrzeze (find_running() patrzy na WSZYSTKIE warianty).
            input(DIM("  [Enter]"))
        elif ch == "k":
            queue_edit(states)
            input(DIM("  [Enter]"))
        elif ch == "x":
            stop()
            input(DIM("  [Enter]"))
        elif ch == "r":
            sc = pick_scene(states, "  scena do REGENERACJI od zera (numer albo nazwa): ")
            if sc:
                regenerate(sc)
            input(DIM("  [Enter]"))
        elif ch == "v":
            sc = pick_scene(states)
            if sc:
                print()
                run_gen("--verify", sc)
            input(DIM("\n  [Enter]"))
        elif ch == "t":
            table(states)
            input(DIM("  [Enter]"))
        elif ch == "d":
            sc = pick_scene(states)
            if sc:
                print()
                run_gen("--dry-run", "--scene", sc)
            input(DIM("\n  [Enter]"))


def switch_variant(name):
    """Przelacza wariant w tym module i w echo_core. -> True, jesli sie udalo.

    Zmiana dotyczy tylko WIDOKU i tego, co uruchomia kolejne polecenia — nie ma
    zadnego wplywu na proces generacji, ktory juz dziala (ma wlasny, zamrozony
    w chwili startu zestaw sciezek).
    """
    global VARIANT
    try:
        G.set_variant(name)
    except ValueError as e:
        print(RED(f"  {e}"))
        return False
    VARIANT = name
    return True


def apply_variant(args):
    """Wyjmuje --variant/-v z argv, ustawia wariant w tym modulu i w echo_core.

    Musi zadzialac PRZED czymkolwiek, co dotyka sciezek — P.OUT_ROOT i mesh sceny
    zmieniaja sie wraz z wariantem. Zwraca argv bez zjedzonych argumentow.
    """
    global VARIANT
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("--variant", "-v") and i + 1 < len(args):
            VARIANT = args[i + 1]
            i += 2
            continue
        if a.startswith("--variant="):
            VARIANT = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if not switch_variant(VARIANT):
        raise SystemExit(2)
    return out


def main():
    args = apply_variant(sys.argv[1:])
    if not args:
        return dashboard()
    cmd, rest = args[0], args[1:]
    if cmd == "status":
        states, running_map = all_states()
        print()
        header(states, running_map)
        table(states)
        return 0
    if cmd == "watch":
        return watch(rest[0] if rest else None)
    if cmd == "next":
        states, _ = all_states()
        nxt = next_scene(states)
        if not nxt:
            print(GREEN(f"  Wszystkie {len(SCENES())} scen wariantu {VARIANT} gotowe."))
            return 0
        return start(nxt["scene"])
    if cmd == "_run-queue":          # wewnetrzne: nadzorca kolejki
        return run_queue()
    if cmd in ("queue", "q", "kolejka"):
        states, _ = all_states()
        sub = rest[0] if rest else "list"
        q = read_queue() or {}
        cur = list(q.get("pending") or [])
        if sub == "list":
            # Bez argumentow POKAZUJEMY kolejke, nie kolejkujemy wszystkiego —
            # przypadkowe `queue` nie moze juz wrzucic 17 scen.
            alive = supervisor_alive(q)
            print(f"\n  KOLEJKA — wariant {VARIANT}"
                  + (GREEN("  (nadzorca dziala)") if alive else DIM("  (nadzorca nie dziala)")))
            for i, sc in enumerate(cur, 1):
                print(f"    {i:>2}. {sc}")
            if not cur:
                print(DIM("    — pusta —"))
            if q.get("done"):
                print(DIM(f"    gotowe: {', '.join(q['done'])}"))
            if q.get("failed"):
                print(RED(f"    nieudane: {', '.join(q['failed'])}"))
            print(DIM("\n  queue all              zakolejkuj wszystko, co niegotowe"))
            print(DIM("  queue <sceny...>       ustaw dokladnie te sceny"))
            print(DIM("  queue add <sceny...>   dodaj do kolejki"))
            print(DIM("  queue rm  <sceny...>   usun z kolejki (dziala na zywo)"))
            print(DIM("  queue clear            oproznij kolejke (NIE przerywa biezacej sceny)"))
            print(DIM("  w pulpicie: [k] — lista z przelacznikami\n"))
            return 0
        if sub == "clear":
            return queue_apply([], states)
        if sub == "all":
            return queue_start([], states)
        if sub == "add":
            bad = [x for x in rest[1:] if x not in SCENES()]
            if bad:
                print(RED(f"  nieznane sceny: {', '.join(bad)}"))
                return 1
            want = set(cur) | set(rest[1:])
            return queue_apply([sc for sc in SCENES() if sc in want], states)
        if sub in ("rm", "remove", "del"):
            bad = [x for x in rest[1:] if x not in cur]
            if bad:
                print(YELLOW(f"  nie ma w kolejce: {', '.join(bad)}"))
            return queue_apply([sc for sc in cur if sc not in set(rest[1:])], states)
        return queue_start(rest, states)
    if cmd == "start":
        if not rest:
            print(RED("  Podaj scene: echo_ctl.py start <scena>"))
            return 2
        return start(rest[0])
    if cmd in ("regen", "regenerate"):
        if not rest:
            print(RED("  Podaj scene: echo_ctl.py regen <scena>"))
            return 2
        return regenerate(rest[0])
    if cmd == "stop":
        return stop(rest[0] if rest else None)
    if cmd == "verify":
        if not rest:
            print(RED("  Podaj scene: echo_ctl.py verify <scena>"))
            return 2
        return run_gen("--verify", rest[0])
    if cmd == "dry":
        if not rest:
            print(RED("  Podaj scene: echo_ctl.py dry <scena>"))
            return 2
        return run_gen("--dry-run", "--scene", rest[0])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
