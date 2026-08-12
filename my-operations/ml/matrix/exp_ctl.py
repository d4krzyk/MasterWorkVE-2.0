#!/usr/bin/env python
"""Pulpit kontrolny macierzy eksperymentow -- analogicznie do `echo_ctl.py`.

    python my-operations/ml/exp_ctl.py plan            # co jest do zrobienia i ile potrwa
    python my-operations/ml/exp_ctl.py status          # stan wszystkich przebiegow
    python my-operations/ml/exp_ctl.py next            # co uruchomic jako nastepne
    python my-operations/ml/exp_ctl.py start A 0       # uruchom warunek A, ziarno 0
    python my-operations/ml/exp_ctl.py watch           # podglad na zywo
    python my-operations/ml/exp_ctl.py stop            # zatrzymaj biezacy przebieg
    python my-operations/ml/exp_ctl.py results         # zestawienie metryk

Ta sama zasada, co przy generowaniu danych: NIC nie rusza samo. Uruchomienie
przebiegu jest jawna decyzja czlowieka, bo GPU jest jedno, przebieg trwa
godziny, a kolejnosc warunkow ma znaczenie naukowe (najpierw te, ktore
odpowiadaja na pytanie pracy).

Wykrywanie dzialajacego przebiegu idzie przez skan `/proc` po wierszu polecen,
a nie przez plik PID: po twardym zabiciu procesu plik PID klamie, a `/proc` nie.
Przebiegi startuja przez `start_new_session=True`, wiec zerwane SSH ich nie
zabija -- 17-godzinny przebieg nie moze zalezec od tego, czy laptop nie uspil sie
w nocy.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.matrix"

from .. import paths  # noqa: E402
from .experiments import (  # noqa: E402
    BATCH_SIZE,
    CONDITIONS,
    CONDITIONS_BY_ID,
    GROUPS,
    MODEL_ECHO,
    SEEDS,
    TOTAL_STEPS,
    RunSpec,
    dump_config,
    matrix_summary,
    plan_status,
)
from ..dataset.splits import load_splits  # noqa: E402

# `train_condition.py` mieszka w `depth_model/`, nie obok tego pliku -- po
# reorganizacji z 2026-08-12 `parent` wskazywalby `matrix/` i `start`/`next`
# proponowalyby nieistniejaca sciezke.
TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "depth_model" / "train_condition.py"
assert TRAIN_SCRIPT.exists(), f"nie znaleziono {TRAIN_SCRIPT}"

# Zmierzone na tym sprzecie (RTX 5070 Ti, batch 32, AMP) -- patrz
# outputs/ml/bench/bench_main.json. Sluzy TYLKO do planowania czasu.
SEC_PER_STEP = {
    ("full", False): 1.5135,   # pelny model, nn.Bilinear z torcha
    ("full", True): 0.0776,    # pelny model, BilinearEinsum
    # ECHO2DEPTH jest ograniczony dataloaderem (4.4 ms GPU wobec 12.1 ms I/O),
    # wiec liczy sie ta druga liczba, nie czas GPU.
    (MODEL_ECHO, False): 0.0121,
    (MODEL_ECHO, True): 0.0121,
}


def C(t, code):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t


def sec_per_step(condition, fast_bilinear: bool) -> float:
    return SEC_PER_STEP[(condition.model if condition.model == MODEL_ECHO else "full",
                         fast_bilinear)]


def find_running() -> dict[str, int]:
    """run_id -> PID dla kazdego dzialajacego `train_condition.py`."""
    out: dict[str, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        args = [a.decode("utf-8", "replace") for a in argv if a]
        if not any(a.endswith("train_condition.py") for a in args):
            continue
        cond = seed = None
        for i, a in enumerate(args):
            if a == "--condition" and i + 1 < len(args):
                cond = args[i + 1]
            elif a == "--seed" and i + 1 < len(args):
                seed = args[i + 1]
        if cond and seed is not None:
            out[f"{cond}_seed{seed}"] = int(entry.name)
    return out


def _block(rec: dict, name: str) -> dict:
    """Blok metryk z rekordu walidacji, niezaleznie od WERSJI formatu.

    Do 2026-08-11 `evaluate()` zwracalo `overall`/`edge`/`smooth`; po przejsciu
    na tabele statystyk per probka klucz `overall` nazywa sie `all`. Na dysku
    leza pliki OBU formatow -- przebiegi sprzed i po zmianie protokolu -- wiec
    czytnik musi znac oba. Wczesniej `["overall"]` rzucalo `KeyError`, ktory byl
    po cichu polykany przez `except`: dla DZIALAJACEGO przebiegu kolumna
    `best RMSE` pokazywala wtedy `-`, mimo ze wynik byl juz w `metrics.jsonl`.
    """
    return rec.get(name) or rec.get({"overall": "all", "all": "overall"}.get(name, name)) or {}


def _rmse_of(rec: dict) -> float:
    return _block(rec, "overall")["RMSE"]


def run_state(cond_id: str, seed: int, running: dict) -> dict:
    spec = RunSpec(condition=cond_id, seed=seed)
    d = spec.run_dir()
    st = {"run_id": spec.run_id, "condition": cond_id, "seed": seed,
          "dir": d, "exists": d.exists(), "pid": running.get(spec.run_id),
          "step": 0, "total": TOTAL_STEPS, "best_rmse": None,
          "finished": False, "state": "nie zaczety", "val_subset": None}
    if not d.exists():
        return st

    sfp = d / "status.json"
    if sfp.exists():
        try:
            s = json.loads(sfp.read_text(encoding="utf-8"))
            st.update(step=s.get("step", 0), total=s.get("total_steps", TOTAL_STEPS),
                      best_rmse=s.get("best_val_rmse"), finished=s.get("finished", False),
                      val_subset=s.get("val_angle_subset", "<sprzed 2026-08-11>"))
        except (json.JSONDecodeError, OSError):
            pass

    # `metrics.jsonl` jest dopisywany na biezaco, wiec dla DZIALAJACEGO przebiegu
    # jest swiezszy niz `status.json` (ten powstaje dopiero na koncu).
    #
    # Ale tylko dla dzialajacego. Gdy `status.json` juz istnieje, jest ZRODLEM
    # PRAWDY i nie wolno go "poprawiac" minimum z pliku logu: przy ponownym
    # uruchomieniu z `--force` log potrafil zawierac wpisy dwoch przebiegow
    # naraz, a minimum po calosci pokazywalo wynik tego STARSZEGO. Od 2026-08-12
    # `train_condition.py` kasuje logi na swiezym starcie, ale stare katalogi na
    # dysku nadal moga byc zanieczyszczone -- wiec czytnik tez musi byc odporny.
    mfp = d / "metrics.jsonl"
    if mfp.exists():
        try:
            lines = [ln for ln in mfp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                st["step"] = max(st["step"], last.get("step", 0))
                if st["best_rmse"] is None:
                    st["best_rmse"] = min(_rmse_of(json.loads(ln)) for ln in lines)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    if st["pid"]:
        st["state"] = "DZIALA"
    elif st["finished"] or st["step"] >= st["total"]:
        st["state"] = "gotowy"
        st["finished"] = True
    elif st["step"] > 0:
        st["state"] = "przerwany"
    else:
        st["state"] = "pusty"
    return st


def all_states(groups=None, seeds=SEEDS) -> list[dict]:
    running = find_running()
    out = []
    for c in CONDITIONS:
        if groups and c.group not in groups:
            continue
        for s in seeds:
            st = run_state(c.id, s, running)
            st["group"] = c.group
            st["model"] = c.model
            st["geometry"] = c.geometry
            st["angle_subset"] = c.angle_subset
            out.append(st)
    return out


def bar(frac, width=20):
    n = int(max(0.0, min(1.0, frac)) * width)
    return "[" + "#" * n + "." * (width - n) + "]"


def fmt_h(h):
    return f"{h:5.2f} h" if h >= 0.1 else f"{h*60:5.1f} min"


# --------------------------------------------------------------------- komendy


def cmd_plan(args) -> int:
    groups = args.groups.split(",") if args.groups else list(GROUPS)
    splits_cache = {}
    print("=" * 100)
    print(f"PLAN MACIERZY   {TOTAL_STEPS} krokow/przebieg, batch {BATCH_SIZE}, "
          f"ziarna {list(SEEDS)}, fast_bilinear={args.fast_bilinear}")
    print("=" * 100)
    print(f"{'id':<5}{'grupa':<15}{'subset':<16}{'kat':>4}  {'model':<11}{'geom':<9}"
          f"{'train':>8}{'ep.rown':>9}{'h/przeb':>10}{'h x3':>9}  izoluje")
    print("-" * 105)
    tot = 0.0
    for c in CONDITIONS:
        if c.group not in groups:
            continue
        sp = splits_cache.setdefault(c.geometry, load_splits(variant=c.geometry))
        n = c.n_train_samples(sp)
        sps = sec_per_step(c, args.fast_bilinear)
        h = TOTAL_STEPS * sps / 3600
        tot += h * len(SEEDS)
        from ..dataset import angles as A
        print(f"{c.id:<5}{c.group:<15}{c.angle_subset:<16}"
              f"{A.angles_per_location(c.angle_subset):>4}  {c.model:<11}{c.geometry:<9}"
              f"{n:>8}{c.epochs_equivalent(sp):>9.1f}{h:>10.2f}{h*len(SEEDS):>9.2f}  {c.isolates}")
    print("-" * 105)
    n_runs = sum(len(SEEDS) for c in CONDITIONS if c.group in groups)
    print(f"RAZEM: {n_runs} przebiegow, {tot:.1f} h = {tot/24:.2f} dni GPU")
    if not args.fast_bilinear:
        alt = sum(TOTAL_STEPS * sec_per_step(c, True) / 3600 * len(SEEDS)
                  for c in CONDITIONS if c.group in groups)
        print(f"       z --fast-bilinear byloby {alt:.1f} h = {alt/24:.2f} dni "
              f"({tot/alt:.1f}x szybciej, ta sama funkcja matematyczna)")
    fp = dump_config(s_per_step=sec_per_step(CONDITIONS_BY_ID["B"], args.fast_bilinear))
    print(f"\nkonfiguracja zapisana: {fp}")
    return 0


def cmd_status(args) -> int:
    groups = args.groups.split(",") if args.groups else None
    states = all_states(groups)
    running = [s for s in states if s["state"] == "DZIALA"]
    print("=" * 97)
    # Postep liczony wobec PLANU, nie wobec przestrzeni projektowej. "14/66"
    # czyta sie jako 21 %, podczas gdy 35 z tych 66 nigdy nie mialo pojsc.
    in_plan = [s for s in states if plan_status(s["condition"], s["seed"])[0]]
    done_plan = sum(1 for s in in_plan if s["finished"])
    done_all = sum(1 for s in states if s["finished"])
    print(f"STATUS   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   "
          f"dziala: {len(running)}   "
          f"gotowych: {done_plan}/{len(in_plan)} wg PLANU"
          f"   ({done_all}/{len(states)} calej przestrzeni projektowej)")
    print("=" * 97)
    print(f"{'run_id':<14}{'grupa':<15}{'stan':<11}{'postep':<24}{'krok':>13}"
          f"{'best val RMSE':>15}{'prot':>6}{'plan':>6}")
    print("-" * 109)
    stale = 0
    for s in states:
        col = {"DZIALA": 32, "gotowy": 36, "przerwany": 33}.get(s["state"], 90)
        rmse = f"{s['best_rmse']:.5f}" if s["best_rmse"] is not None else "-"
        state_cell = C(f"{s['state']:<11}", col)
        progress = bar(s["step"] / max(1, s["total"]))
        # Protokol walidacji: 'all' = po decyzji 2026-08-11 §1 (36 katow),
        # cokolwiek innego = checkpoint sprzed niej, NIEPOROWNYWALNY z reszta.
        prot = s.get("val_subset")
        if prot is None:
            cell = ""
        elif prot == "all":
            cell = "36"
        else:
            cell = C("STARY", 31)
            stale += 1
        in_plan, _ = plan_status(s["condition"], s["seed"])
        plan_cell = "tak" if in_plan else C("—", 90)
        print(f"{s['run_id']:<14}{s['group']:<15}{state_cell}"
              f"{progress:<24}{s['step']:>6}/{s['total']:<6}{rmse:>15}{cell:>6}{plan_cell:>6}")
    print("-" * 109)
    # `best val RMSE` to WALIDACJA, nie test -- liczby do pracy daje `evaluate.py`
    # na `test@36`. Bez tej adnotacji latwo zestawic ze soba dwie rozne wielkosci.
    print(C("  kolumna 'best val RMSE' to walidacja (val@36), NIE zbior testowy; "
              "liczby do pracy: evaluate.py --run-dir ...", 90))
    if stale:
        print(C(f"  UWAGA: {stale} przebieg(ow) sprzed zmiany protokolu walidacji "
                f"(2026-08-11 §1) -- nieporownywalne z reszta bez przeliczenia.", 31))

    # Kolumna `plan` = "—" nie znaczy "zapomniane". Kazdy taki przebieg ma powod,
    # ktory tu wypisujemy -- inaczej za miesiac nikt nie odtworzy, czemu 35 z 66
    # nie poszlo.
    powody: dict[str, list[str]] = {}
    for s in states:
        ok, why = plan_status(s["condition"], s["seed"])
        if not ok:
            powody.setdefault(why, []).append(s["run_id"])
    if powody:
        print(C(f"\n  POZA PLANEM ({sum(len(v) for v in powody.values())} przebiegow) "
                f"— nie zapomniane, tylko odwolane albo odsuniete:", 90))
        for why, runs in sorted(powody.items(), key=lambda x: -len(x[1])):
            print(C(f"    {len(runs):2d}x  {', '.join(runs[:6])}"
                    f"{' ...' if len(runs) > 6 else ''}", 90))
            print(C(f"         powod: {why}", 90))
    for s in running:
        print(f"  PID {s['pid']}  {s['run_id']}  -> {s['dir']}")
    return 0


def cmd_next(args) -> int:
    """Sugeruje nastepny przebieg wg priorytetu naukowego, nie alfabetycznie."""
    states = all_states()
    running = [s for s in states if s["state"] == "DZIALA"]
    if running:
        print(f"UWAGA: juz dziala {len(running)} przebieg(ow) -- GPU jest jedno.")
        for s in running:
            print(f"  {s['run_id']} (PID {s['pid']}, krok {s['step']}/{s['total']})")
        print()
    order = {g: i for i, g in enumerate(GROUPS)}
    # Tylko przebiegi Z PLANU. Bez tego `next` proponowal np. `SE --seed 1`,
    # odwolane decyzja z 2026-08-11 §2 -- czyli podpowiadal robote, ktora zapadla
    # decyzja, zeby jej nie robic.
    todo = [s for s in states if not s["finished"] and s["state"] != "DZIALA"
            and plan_status(s["condition"], s["seed"])[0]]
    todo.sort(key=lambda s: (order.get(s["group"], 99), s["seed"], s["condition"]))
    if not todo:
        print("Wszystko gotowe.")
        return 0
    print("Kolejnosc sugerowana. Grupa 'bramka' idzie PIERWSZA: mierzy calkowity wklad\necha, czyli GORNE OGRANICZENIE na efekt gestosci. Jesli wyjdzie znikomy, macierz na\npelnym modelu nie jest w stanie wykryc efektu i ciezar dowodu idzie na 'echo' i Model 2.")
    for s in todo[:10]:
        flag = " [wznowic --resume]" if s["state"] == "przerwany" else ""
        print(f"  python {TRAIN_SCRIPT.relative_to(paths.REPO_ROOT)} "
              f"--condition {s['condition']} --seed {s['seed']}{flag}")
    print(f"\n... lacznie {len(todo)} przebiegow do zrobienia")
    return 0


def cmd_start(args) -> int:
    cond, seed = args.condition, args.seed
    if cond not in CONDITIONS_BY_ID:
        print(f"nieznany warunek {cond!r}; dostepne: {sorted(CONDITIONS_BY_ID)}")
        return 2
    running = find_running()
    if running and not args.allow_parallel:
        print("GPU jest jedno, a juz cos dziala:")
        for k, pid in running.items():
            print(f"  {k} (PID {pid})")
        print("Uzyj --allow-parallel, jesli naprawde tego chcesz (ryzyko OOM).")
        return 3

    spec = RunSpec(condition=cond, seed=seed)
    d = spec.run_dir()
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--condition", cond, "--seed", str(seed)]
    if args.fast_bilinear:
        cmd.append("--fast-bilinear")
    if args.resume:
        cmd.append("--resume")
    if args.extra:
        cmd += args.extra.split()

    log = d / "stdout.txt"
    print(f"start: {' '.join(cmd)}")
    print(f"log  : {log}")
    if args.dry_run:
        print("--dry-run: nie uruchomiono")
        return 0
    with log.open("ab") as fh:
        # start_new_session: przebieg przezyje zamkniecie terminala/SSH.
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True,
                             cwd=str(paths.REPO_ROOT))
    print(f"PID {p.pid}")
    return 0


def cmd_stop(args) -> int:
    running = find_running()
    if not running:
        print("nic nie dziala")
        return 0
    for rid, pid in running.items():
        if args.run_id and rid != args.run_id:
            continue
        # SIGTERM, nie SIGKILL: `train_condition.py` przechwytuje go i zapisuje
        # checkpoint, wiec przebieg da sie wznowic bez straty godzin.
        print(f"SIGTERM -> {rid} (PID {pid}); zapisze checkpoint i zakonczy")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print("  proces juz nie istnieje")
    return 0


def cmd_watch(args) -> int:
    try:
        while True:
            os.system("clear")
            cmd_status(args)
            print(f"\n(odswiezanie co {args.interval}s, Ctrl-C konczy)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_results(args) -> int:
    """Zestawienie metryk wszystkich gotowych przebiegow."""
    states = all_states()
    rows = []
    for s in states:
        bfp = s["dir"] / "best.json"
        if not bfp.exists():
            continue
        try:
            b = json.loads(bfp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append((s, b))
    if not rows:
        print("Brak gotowych wynikow (zaden przebieg nie zapisal best.json).")
        return 0

    print("=" * 104)
    print("WYNIKI (najlepszy checkpoint wg RMSE walidacyjnego)")
    print("=" * 104)
    print(f"{'run_id':<14}{'subset':<16}{'krok':>7}{'RMSE':>9}{'krawedz':>9}{'gladkie':>9}"
          f"{'REL':>8}{'log10':>8}{'d1':>8}{'%px kraw':>10}")
    print("-" * 104)
    for s, b in sorted(rows, key=lambda r: (r[0]["group"], r[0]["condition"], r[0]["seed"])):
        o, e, sm = _block(b, "overall"), _block(b, "edge"), _block(b, "smooth")
        print(f"{s['run_id']:<14}{s['angle_subset']:<11}{b['step']:>7}"
              f"{o['RMSE']:>9.4f}{e['RMSE']:>9.4f}{sm['RMSE']:>9.4f}"
              f"{o['ABS_REL']:>8.4f}{o['LOG10']:>8.4f}{o['DELTA1']:>8.4f}"
              f"{b['edge_pixel_fraction']*100:>9.1f}%")
    print("-" * 104)

    if args.per_scene:
        print("\nrozbicie per scena held-out (RMSE):")
        for s, b in sorted(rows, key=lambda r: r[0]["run_id"]):
            per = b.get("per_scene", {})
            cells = "  ".join(f"{k}={v['RMSE']:.4f}" for k, v in sorted(per.items()))
            print(f"  {s['run_id']:<14} {cells}")

    out = paths.ML_OUTPUTS / "results_summary.json"
    out.write_text(json.dumps([{"run": s["run_id"], **b} for s, b in rows],
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nzapisano: {out}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--groups", default=None,
                        help=f"filtr grup po przecinku, z {GROUPS}")
        return sp

    pl = common(sub.add_parser("plan", help="co jest do zrobienia i ile potrwa"))
    pl.add_argument("--fast-bilinear", action="store_true",
                    help="licz czasy przy szybkim zamienniku nn.Bilinear")
    pl.set_defaults(func=cmd_plan)

    st = common(sub.add_parser("status", help="stan wszystkich przebiegow"))
    st.set_defaults(func=cmd_status)

    nx = sub.add_parser("next", help="co uruchomic jako nastepne")
    nx.set_defaults(func=cmd_next, groups=None)

    sr = sub.add_parser("start", help="uruchom jeden przebieg")
    sr.add_argument("condition")
    sr.add_argument("seed", type=int)
    sr.add_argument("--fast-bilinear", action="store_true")
    sr.add_argument("--resume", action="store_true")
    sr.add_argument("--allow-parallel", action="store_true")
    sr.add_argument("--dry-run", action="store_true")
    sr.add_argument("--extra", default=None, help="dodatkowe argumenty do train_condition.py")
    sr.set_defaults(func=cmd_start, groups=None)

    sp_ = sub.add_parser("stop", help="zatrzymaj (SIGTERM -> checkpoint)")
    sp_.add_argument("run_id", nargs="?", default=None)
    sp_.set_defaults(func=cmd_stop, groups=None)

    w = common(sub.add_parser("watch", help="podglad na zywo"))
    w.add_argument("--interval", type=int, default=20)
    w.set_defaults(func=cmd_watch)

    rs = sub.add_parser("results", help="zestawienie metryk")
    rs.add_argument("--per-scene", action="store_true")
    rs.set_defaults(func=cmd_results, groups=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
