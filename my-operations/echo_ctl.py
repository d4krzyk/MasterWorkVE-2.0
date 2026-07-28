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
    python my-operations/echo_ctl.py stop
    python my-operations/echo_ctl.py verify <scena>

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

GEN_SCRIPT = G.SCRIPT_PATH
SCENE_INDEX_CACHE = G.OUT_ROOT / ".scene_index.json"

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
        scene = None
        for i, a in enumerate(argv):
            if a == "--scene" and i + 1 < len(argv):
                scene = argv[i + 1]
        out.append((int(entry.name), scene))
    return out


def scene_index():
    """scena -> liczba lokalizacji. Cache na dysku, bo liczenie wymaga
    wczytania 913 MB pkl (0.9 s) — za duzo jak na widok odswiezany co 2 s."""
    if SCENE_INDEX_CACHE.exists():
        try:
            return json.loads(SCENE_INDEX_CACHE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    idx = {}
    for s in G.SCENE_ORDER:
        try:
            idx[s] = len(G.load_scene_locations(s)[0])
        except Exception:
            idx[s] = 0
    SCENE_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCENE_INDEX_CACHE.write_text(json.dumps(idx, indent=2))
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
    running_map = {s: p for p, s in find_running() if s}
    return [scene_state(s, index, running_map) for s in G.SCENE_ORDER], running_map


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
    print(BOLD("  Visual Echoes 2.0 · generator ech 36-orientacyjnych"))
    print(DIM("  " + "─" * (width - 2)))

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
    n_note = (f"N zmierzone dla {len(n_seen)}/18 scen, reszta {G.MEAN_N_SPEC} wg spec"
              if n_seen else f"N {G.MEAN_N_SPEC} wg spec")
    print(f"  sceny {BOLD(f'{done}/18')} · próbki {written}/{expected} "
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


def start(scene, states=None):
    running = find_running()
    if running:
        pid, sc = running[0]
        print(RED(f"  Generacja juz dziala: {sc} (pid {pid})."))
        print(DIM("  Jeden Simulator na raz — rownolegle sceny biłyby sie o GPU."))
        return 1
    if scene not in G.SCENE_ORDER:
        print(RED(f"  Nieznana scena: {scene}"))
        return 1

    partial = G.scene_h5(scene).exists()
    cmd = [sys.executable, str(GEN_SCRIPT), "--scene", scene]
    if partial:
        cmd.append("--resume")

    G.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = G.OUT_ROOT / f"{scene}.stdout"
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
    running = find_running()
    if not running:
        print(DIM("  Nic nie dziala."))
        return 0
    for pid, sc in running:
        if scene and sc != scene:
            continue
        try:
            os.kill(pid, signal.SIGINT)
            print(YELLOW(f"  SIGINT -> {sc} (pid {pid})"))
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
    return subprocess.call([sys.executable, str(GEN_SCRIPT), *extra], cwd=str(G.REPO_ROOT))


def pick_scene(states, prompt="  scena (numer albo nazwa): "):
    table(states)
    raw = input(prompt).strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(G.SCENE_ORDER):
        return G.SCENE_ORDER[int(raw) - 1]
    if raw in G.SCENE_ORDER:
        return raw
    print(RED(f"  Nieznana scena: {raw}"))
    return None


# ---------------------------------------------------------------------------
# Pulpit
# ---------------------------------------------------------------------------
MENU = """
  {p} podgląd na żywo          {v} weryfikuj scenę
  {n} uruchom następną         {t} tabela wszystkich scen
  {s} uruchom wybraną          {d} próbny przebieg (dry-run)
  {x} zatrzymaj generację      {q} wyjście
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
        print(MENU.format(p=BOLD("p"), n=BOLD("n"), s=BOLD("s"), x=BOLD("x"),
                          v=BOLD("v"), t=BOLD("t"), d=BOLD("d"), q=BOLD("q")))
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
                print(GREEN("  Wszystkie 18 scen gotowe."))
            input(DIM("  [Enter]"))
        elif ch == "s":
            sc = pick_scene(states)
            if sc:
                start(sc)
            input(DIM("  [Enter]"))
        elif ch == "x":
            stop()
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


def main():
    args = sys.argv[1:]
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
            print(GREEN("  Wszystkie 18 scen gotowe."))
            return 0
        return start(nxt["scene"])
    if cmd == "start":
        if not rest:
            print(RED("  Podaj scene: echo_ctl.py start <scena>"))
            return 2
        return start(rest[0])
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
