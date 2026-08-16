#!/usr/bin/env python
"""Kolejka nocna fazy ML — jedno wywolanie liczy wszystko bez nadzoru.

    python my-operations/ml/matrix/ml_ctl.py plan     # co poszloby, ile godzin, ile GB
    python my-operations/ml/matrix/ml_ctl.py run      # uruchom kolejke
    python my-operations/ml/matrix/ml_ctl.py status   # stan po fakcie / w trakcie

KONWENCJA PRZEJETA Z `echo_ctl.py` (pulpit generowania zbioru), swiadomie, zeby
faza ML dzialala tak samo jak faza generowania:

  * plik kolejki JSON z `pending` / `done` / `failed` / `current`, przezywa
    restart procesu i pozwala wznowic bez liczenia czegokolwiek drugi raz;
  * wykrywanie juz dzialajacego przebiegu przez SKAN /proc po wierszu polecenia,
    a nie przez plik PID -- ten po twardym zabiciu klamie;
  * log per krok w osobnym pliku + zbiorcze podsumowanie na koncu;
  * znaczniki czasu `[HH:MM:SS]` w kazdej linii nadzorcy;
  * `plan` pokazuje pelny zamiar bez uruchamiania czegokolwiek;
  * potomek startowany z `start_new_session=True`, zeby zerwane SSH go nie zabilo.

JEDNA SWIADOMA ROZNICA WOBEC `echo_ctl.py`. Tam druga nieudana proba PRZERYWA
cala kolejke -- slusznie, bo typowa przyczyna to zawieszony GPU, a kolejne sceny
tylko dopisywalyby bledy przez cala noc. Tutaj krok, ktory padl, jest logowany
i kolejka IDZIE DALEJ: kroki 3-5 (geometria, glowne, krzywa) nie zaleza od
Modelu 2, wiec jego niepowodzenie nie moze skasowac reszty nocy. Awaria samego
GPU i tak zatrzyma wszystko przez kontrole wolnego miejsca i kolejne niezerowe
kody wyjscia, ktore trafia do podsumowania.

CZEGO Z `echo_ctl.py` NIE PRZENIESIONO i dlaczego:
  * interaktywne menu (`watch`, klawisze s/d/q) -- kolejka ma chodzic bez
    czlowieka przy klawiaturze, a stan pokazuje `status`;
  * `verify` (kontrola kompletnosci HDF5) -- to wlasnosc zbioru danych,
    sprawdzana przez `dataset/echo_data.py --verify-loader`, nie przez trening;
  * ponawianie kroku -- przebieg treningowy wznawia sie z `--resume` sam
    z siebie, a ponawianie w petli maskowaloby prawdziwa przyczyne awarii.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.matrix"

from .. import paths  # noqa: E402
from .experiments import PARAM_COUNTS, SEC_PER_STEP, TOTAL_STEPS  # noqa: E402

ML = paths.REPO_ROOT / "my-operations" / "ml"
TRAIN = ML / "depth_model" / "train_condition.py"
EVAL = ML / "depth_model" / "evaluate.py"
PRETRAIN = ML / "pretext_model" / "train_pretext.py"
TRANSFER = ML / "pretext_model" / "transfer.py"
PRETEXT_SUM = ML / "pretext_model" / "summarize.py"
NUMBERS = ML / "analysis" / "thesis_numbers.py"
FINAL = ML / "analysis" / "final_results.py"

LOGS = paths.ML_OUTPUTS / "logs"
QUEUE = paths.ML_OUTPUTS / "logs" / "ml_ctl_queue.json"

# Katalog logow powstaje JUZ PRZY IMPORCIE, a nie dopiero przy pierwszym kroku.
# Powod praktyczny: typowe uruchomienie na noc to
#     nohup python ... ml_ctl.py run > outputs/ml/logs/ml_ctl_nohup.log 2>&1 &
# a przekierowanie powloki wykonuje sie ZANIM Python w ogole wystartuje. Gdy
# katalogu nie ma, bash zglasza blad, potomek konczy sie natychmiast i kolejka
# NIE RUSZA -- przy czym `[1] <pid>` w terminalu wyglada, jakby ruszyla.
LOGS.mkdir(parents=True, exist_ok=True)

# Wyjscie nadzorcy linia po linii, nawet gdy idzie do PLIKU. Bez tego Python
# buforuje stdout blokowo przy przekierowaniu, wiec `tail -f` na logu nocnej
# kolejki nie pokazuje nic przez godziny -- a to jedyny sposob, zeby sprawdzic,
# czy cos sie dzieje, bez zabijania procesu.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):   # nie-TextIO albo starszy Python
    pass

# Ponizej tego progu kolejka sie ZATRZYMUJE, zamiast zapisywac na pelny dysk.
MIN_FREE_GB = 15.0

PRETEXT_K = (4, 12, 36)
PRETEXT_SUBSAMPLE = 16          # kontrola: K=36 podprobkowane do 16 par/lokalizacje

# Przepustowosc ZMIERZONA 2026-08-12 na tym sprzecie (batch 32, 120 krokow):
#   pretekst K=4  1651 par/s | K=36 1384 par/s | transfer 2019 probek/s
# Przeliczone na godziny za 40 000 krokow, z zapasem 15 % na walidacje.
H_PRETEXT = {4: 0.25, 12: 0.27, 36: 0.30}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def free_gb() -> float:
    return shutil.disk_usage(paths.OUTPUTS).free / 1024 ** 3


# --------------------------------------------------------------------- kroki


def _pretext_dir(k: int, sub: int | None, seed: int) -> Path:
    tag = f"K{k}" + (f"_p{sub}" if sub else "")
    return paths.ML_OUTPUTS / "pretext" / f"pretext_{tag}_seed{seed}"


# Warunki `geometria_echo` i ich odpowiedniki w `main` -- do ewaluacji z maskami.
MASK_PAIRS = (("EPA", "EA"), ("EPB", "EB"), ("EPD", "ED"))
MASK_MODES = ("intersection", "strict")


def build_steps() -> list[dict]:
    """Kolejka sesji 2026-08-16 (domkniecie po sesji 2026-08-15).

    Plany poprzednich nocy sa w `outputs/ml/logs/ml_ctl_*.md` -- wszystkie ich
    kroki sa gotowe, wiec powtarzanie ich tutaj tylko wydluzaloby `plan`.

    DWIE RZECZY, OBIE DOMYKAJACE ZASTRZEZENIA, A NIE OTWIERAJACE OSIE:

    1. Rozrzut po ziarnach dla ZADANIA PRETEKSTOWEGO. MAAE 25,13 stopnia jest
       liczba naglowkowa ("zadanie jest wykonalne, spadek o 72 % wobec
       losowego"), a ma n = 1 -- jako jedyny wynik tej rangi w calej pracy.
       Wariant `@16par` idzie razem z pozostalymi, bo bez niego rozklad efektu
       (-36,64 stopnia z tytulu liczby par) dalej mialby jeden ze swoich dwoch
       skladnikow bez przedzialu.
    2. Analiza MASEK na ziarnach 1-2. Tabela z 2026-08-13 §3.1 byla liczona na
       ziarnie 0; skoro `EPA - EA` zmienia znak po dolozeniu ziaren, jej wniosek
       o odpornosci ZNAKU trzeba przeliczyc, a nie tylko zawezic komentarzem.
    """
    S: list[dict] = []
    gb_echo = 5 * PARAM_COUNTS["echo2depth"] * 4 / 1024 ** 3

    # --- 1. Pretrening Modelu 2, ziarna 1-2 --------------------------------
    for s in (1, 2):
        for k in PRETEXT_K:
            S.append({"id": f"pretext_K{k}_s{s}", "grupa": "1. pretekst, ziarna 1-2 (rozrzut MAAE)",
                      "cmd": [sys.executable, str(PRETRAIN), "--k", str(k), "--seed", str(s)],
                      "done_marker": str(_pretext_dir(k, None, s) / "status.json"),
                      "godzin": H_PRETEXT[k], "gb": round(gb_echo, 3)})
        S.append({"id": f"pretext_K36_p{PRETEXT_SUBSAMPLE}_s{s}",
                  "grupa": "1. pretekst, ziarna 1-2 (rozrzut MAAE)",
                  "cmd": [sys.executable, str(PRETRAIN), "--k", "36", "--seed", str(s),
                          "--pairs-per-location", str(PRETEXT_SUBSAMPLE)],
                  "done_marker": str(_pretext_dir(36, PRETEXT_SUBSAMPLE, s) / "status.json"),
                  "godzin": H_PRETEXT[36], "gb": round(gb_echo, 3)})

    # --- 2. Ewaluacja z maskami na ziarnach 1-2 ----------------------------
    # Checkpointy juz sa (sesja 2026-08-15), wiec to sama ewaluacja: ~20 s na
    # przebieg. `--intersection-mask` wymaga OBU wariantow geometrii, wiec
    # liczy sie osobno dla `main` i dla `patched`.
    for s in (1, 2):
        for patched, main in MASK_PAIRS:
            for cond in (patched, main):
                for mode in MASK_MODES:
                    d = paths.ML_OUTPUTS / "eval" / f"{cond}_seed{s}_mask-{mode}"
                    S.append({
                        "id": f"maska_{cond}_s{s}_{mode}",
                        "grupa": "2. maski przeciecia i scisla, ziarna 1-2",
                        "cmd": [sys.executable, str(EVAL),
                                "--run-dir", str(paths.RUNS_DIR / f"{cond}_seed{s}"),
                                "--intersection-mask", "--mask-mode", mode],
                        "done_marker": str(d / "eval.json"), "marker_wystarczy": True,
                        "godzin": 0.01, "gb": 0.0})
    return S


# ------------------------------------------------------------------- kolejka


def read_queue() -> dict | None:
    if not QUEUE.exists():
        return None
    try:
        return json.loads(QUEUE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_queue(q: dict) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")


def step_done(step: dict) -> bool:
    """Krok jest gotowy, jesli jego `status.json` istnieje i ma `finished: true`.
    Tak samo jak `echo_ctl.scene_complete()` patrzy na atrybut w HDF5, a nie na
    wlasna ksiegowosc -- zrodlem prawdy jest artefakt, nie plik kolejki.

    WYJATEK `marker_wystarczy`: kroki EWALUACYJNE nie pisza `status.json`, tylko
    `eval.json` -- i pisza go dopiero na koncu, wiec samo istnienie pliku jest
    juz dowodem ukonczenia. Bez tego wyjatku ewaluacje przeliczalyby sie przy
    kazdym wznowieniu kolejki, bo `.get("finished")` zwracalby None.
    """
    m = step.get("done_marker")
    if not m:
        return False
    p = Path(m)
    if not p.exists():
        return False
    if step.get("marker_wystarczy"):
        return True
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("finished"))
    except json.JSONDecodeError:
        return False


def find_running() -> list[tuple[int, str]]:
    """Skan /proc po wierszu polecenia -- jak w `echo_ctl.find_running()`.
    `pgrep -f` dopasowalby takze nasz wlasny proces nadzorcy."""
    out = []
    me = str(Path(__file__).name)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        line = " ".join(a.decode(errors="replace") for a in argv if a)
        if me in line:
            continue
        for script in (TRAIN.name, PRETRAIN.name, TRANSFER.name):
            if script in line:
                out.append((int(entry.name), line[:110]))
                break
    return out


# --------------------------------------------------------------- podkomendy


def cmd_plan(args) -> int:
    steps = build_steps()
    print("=" * 96)
    print(f"PLAN KOLEJKI   {len(steps)} krokow   (nic nie zostalo uruchomione)")
    print("=" * 96)
    grp: dict[str, list[dict]] = {}
    for s in steps:
        grp.setdefault(s["grupa"], []).append(s)
    tot_h = tot_gb = 0.0
    for g, items in grp.items():
        gh = sum(i["godzin"] for i in items)
        gg = sum(i["gb"] for i in items if not step_done(i))
        pend = [i for i in items if not step_done(i) and not i.get("zawsze")]
        print(f"\n  {g}")
        print(f"    krokow {len(items):2d} (do zrobienia {len(pend):2d})   "
              f"{gh:5.2f} h   {gg:6.2f} GB")
        for i in items:
            mark = "gotowe" if step_done(i) else ("zawsze" if i.get("zawsze") else "  -   ")
            print(f"      [{mark}] {i['id']}")
        tot_h += sum(i["godzin"] for i in items if not step_done(i))
        tot_gb += gg
    print("\n" + "-" * 96)
    print(f"  RAZEM do zrobienia: {tot_h:.2f} h GPU, {tot_gb:.1f} GB")
    print(f"  wolne miejsce teraz: {free_gb():.1f} GB   (prog zatrzymania: {MIN_FREE_GB} GB)")
    if free_gb() - tot_gb < MIN_FREE_GB:
        print(f"  UWAGA: po calej kolejce zostanie {free_gb() - tot_gb:.1f} GB "
              f"-- ponizej progu, czesc krokow zostanie pominieta")
    run = find_running()
    if run:
        print(f"  UWAGA: cos juz liczy na GPU: {run[0][1]}")
    return 0


def cmd_status(args) -> int:
    q = read_queue()
    steps = build_steps()
    done = [s["id"] for s in steps if step_done(s)]
    print(f"gotowe {len(done)}/{len(steps)}")
    if q:
        print(f"  ostatni zapis kolejki: {q.get('updated', '?')}")
        print(f"  biezacy: {q.get('current') or '-'}")
        if q.get("failed"):
            print(f"  NIEUDANE: {', '.join(q['failed'])}")
    run = find_running()
    for pid, line in run:
        print(f"  DZIALA pid {pid}: {line}")
    todo = [s["id"] for s in steps if not step_done(s) and not s.get("zawsze")]
    print(f"  do zrobienia ({len(todo)}): {', '.join(todo[:8])}{' ...' if len(todo) > 8 else ''}")
    return 0


def _run_one(step: dict, logf: Path) -> int:
    logf.parent.mkdir(parents=True, exist_ok=True)
    with logf.open("ab") as fh:
        fh.write(f"\n=== {datetime.now().isoformat()} :: {' '.join(step['cmd'])}\n".encode())
        fh.flush()
        return subprocess.call(step["cmd"], stdout=fh, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, cwd=str(paths.REPO_ROOT),
                               start_new_session=True)


def cmd_run(args) -> int:
    steps = build_steps()
    started = datetime.now()
    q = read_queue() or {}
    q.update({"started": started.isoformat(), "done": q.get("done", []),
              "failed": q.get("failed", []), "skipped": q.get("skipped", [])})
    results = []

    print(f"[{_ts()}] KOLEJKA start — {len(steps)} krokow, wolne {free_gb():.1f} GB")
    for step in steps:
        sid = step["id"]
        if step_done(step) and not step.get("zawsze"):
            print(f"[{_ts()}] {sid}: juz gotowe — pomijam")
            if sid not in q["done"]:
                q["done"].append(sid)
            continue

        if free_gb() < MIN_FREE_GB:
            print(f"[{_ts()}] STOP: wolne {free_gb():.1f} GB < prog {MIN_FREE_GB} GB. "
                  f"Kolejka zatrzymana, zeby nie zapisywac na pelny dysk.")
            q["stopped_reason"] = f"wolne miejsce {free_gb():.1f} GB"
            break

        # Krok transferu wymaga enkodera z pretreningu; jesli tamten padl,
        # ten nie ma z czego wystartowac -- pomijamy jawnie, nie wywalamy sie.
        need = step.get("wymaga")
        if need and not Path(need).exists():
            print(f"[{_ts()}] {sid}: POMINIETY — brak {Path(need).name} (pretrening nie doszedl)")
            q["skipped"].append(sid)
            results.append({"id": sid, "wynik": "pominiety", "powod": f"brak {need}"})
            continue

        while find_running():
            print(f"[{_ts()}] czekam: GPU zajete przez {find_running()[0][1]}")
            time.sleep(30)

        q["current"] = sid
        q["updated"] = datetime.now().isoformat()
        write_queue(q)
        logf = LOGS / f"{sid}.log"
        t0 = time.perf_counter()
        print(f"[{_ts()}] start {sid}  ({step['grupa']})")
        rc = _run_one(step, logf)
        dt = (time.perf_counter() - t0) / 60

        if rc == 0:
            print(f"[{_ts()}] {sid} GOTOWE w {dt:.1f} min")
            q["done"].append(sid)
            results.append({"id": sid, "wynik": "ok", "minut": round(dt, 1),
                            "log": str(logf.relative_to(paths.REPO_ROOT))})
            # Ewaluacja od razu po treningu — inaczej rano trzeba by ja robic recznie.
            if step.get("eval"):
                subprocess.call([sys.executable, str(EVAL), "--run-dir", step["eval"]],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=str(paths.REPO_ROOT))
        else:
            # NIE przerywamy kolejki (patrz naglowek modulu): kroki 3-5 nie
            # zaleza od Modelu 2, wiec jego awaria nie moze skasowac reszty nocy.
            print(f"[{_ts()}] {sid} PADL (kod {rc}) po {dt:.1f} min — ide dalej")
            q["failed"].append(sid)
            results.append({"id": sid, "wynik": f"blad {rc}", "minut": round(dt, 1),
                            "log": str(logf.relative_to(paths.REPO_ROOT))})

        q["current"] = None
        q["updated"] = datetime.now().isoformat()
        write_queue(q)

        # `LICZBY_DO_PRACY.md` ma byc aktualne nad ranem niezaleznie od tego,
        # dokad kolejka dojdzie — wiec odswiezamy po KAZDYM kroku, nie na koncu.
        # KOLEJNOSC MA ZNACZENIE: `final_results.py` produkuje plik dowodowy,
        # z ktorego `thesis_numbers.py` czyta -- odwrotnie eksport byloby o krok
        # do tylu przez cala noc.
        for script in (FINAL, NUMBERS):
            subprocess.call([sys.executable, str(script)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, cwd=str(paths.REPO_ROOT))

    _write_summary(results, started, q)
    ok = sum(1 for r in results if r["wynik"] == "ok")
    print(f"[{_ts()}] KOLEJKA koniec — ok {ok}, bledy {len(q['failed'])}, "
          f"pominiete {len(q['skipped'])}, wolne {free_gb():.1f} GB")
    return 0


def _write_summary(results: list[dict], started: datetime, q: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / f"ml_ctl_{started:%Y-%m-%d_%H%M}.md"
    L = [f"# Kolejka ML — {started:%Y-%m-%d %H:%M}", "",
         f"- start: {started:%Y-%m-%d %H:%M:%S}",
         f"- koniec: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"- wolne miejsce na koncu: **{free_gb():.1f} GB**",
         f"- kroki: ok **{sum(1 for r in results if r['wynik'] == 'ok')}**, "
         f"bledy **{len(q.get('failed', []))}**, pominiete **{len(q.get('skipped', []))}**", ""]
    if q.get("stopped_reason"):
        L += [f"> **Kolejka zatrzymana**: {q['stopped_reason']}", ""]
    L += ["| krok | wynik | minut | log |", "|---|---|---|---|"]
    L += [f"| `{r['id']}` | {r['wynik']} | {r.get('minut', '—')} | "
          f"`{r.get('log', '—')}` |" for r in results]
    L += ["", "Liczby odswiezone w `my-operations/docs/LICZBY_DO_PRACY.md` po kazdym kroku.", ""]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[{_ts()}] podsumowanie: {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("plan", help="pokaz plan bez uruchamiania").set_defaults(func=cmd_plan)
    sub.add_parser("run", help="uruchom kolejke").set_defaults(func=cmd_run)
    sub.add_parser("status", help="stan kolejki").set_defaults(func=cmd_status)
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
