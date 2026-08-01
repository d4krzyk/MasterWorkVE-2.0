#!/usr/bin/env python3
"""Generator datasetu ech 36-orientacyjnych dla Visual Echoes 2.0.

Implementuje specyfikacje z `my-operations/docs/GENERATOR_PARAMS.md`. Dokument
jest zrodlem prawdy — kazda stala w `echo_core/params.py` ma tam odwolanie do
eksperymentu, ktory ja rozstrzygnal. Nie zmieniac ich bez zmiany dokumentu.

Ten plik to CLI i dwie petle robocze (generacja pelnego datasetu oraz sam census
sondy). Wszystko wspoldzielone — sciezki, parametry, potok audio, estymatory
szumu, magazyn HDF5, --verify i --status — mieszka w pakiecie `echo_core/`.

Uruchomienie (env `habitat` aktywne):

    conda activate habitat
    python my-operations/generate_echo_dataset.py --dry-run --scene office_1
    python my-operations/generate_echo_dataset.py --scene office_1
    python my-operations/generate_echo_dataset.py --scene office_1 --resume
    python my-operations/generate_echo_dataset.py --probe-only --scene office_1
    python my-operations/generate_echo_dataset.py --verify office_1
    python my-operations/generate_echo_dataset.py --status

`--status`, `--verify`, `--dry-run` NIE dotykaja GPU ani nie tworza Simulatora —
mozna je odpalac w drugim terminalu w trakcie generacji.

Jedna scena = jeden proces OS = jeden dlugo zyjacy Simulator. Powyzej ~30
konstrukcji Simulatora w jednym procesie karta zawiesza sie sprzetowo (procedura
odzysku wymaga prawdziwego resetu PCI), dlatego generator NIGDY nie konstruuje
Simulatora w petli.
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from echo_core.params import (ANGLES_DEG, BYTES_PER_SAMPLE, INDIRECT_RAY_COUNT, MEAN_N_SPEC,
                              N_ANGLES, N_MAX, N_MIN, N_PROBE, S_PER_RENDER_SPEC,
                              SCRIPT_VERSION, SENSOR_HEIGHT, SIGNAL_10DEG, TARGET_SNR,
                              THREAD_COUNT, WARMUP_DISCARD)
from echo_core.paths import (CHIRP_PATH, HABITAT_SIM_PY, LOCATIONS_PKL, MATERIAL_CONFIG,
                             METADATA_ROOT, MY_OPS, OUT_ROOT, PROBE_CENSUS_ROOT, REPO_ROOT,
                             SCENE_ROOT, SCRIPT_PATH, SPEC_DOC, graph_pkl, points_txt,
                             probe_census_csv, probe_census_log, scene_decisions, scene_dir,
                             scene_h5, scene_log, scene_mesh, scene_progress, scene_stdout,
                             scene_verify_dir, set_variant, VARIANTS)
from echo_core.noise import _mean_f32, _rmse, plan_n, sigma_1_from_specs, snr_from_specs
from echo_core.renderer import Renderer
from echo_core.runtime import (_fmt_hms, _install_signal_handlers, setup_logging,
                               interrupted as is_interrupted)
from echo_core.scenes import (HELD_OUT, SCENE_ORDER, load_scene_locations,
                              scenes_for_variant)
from echo_core.status import status
from echo_core.store import DatasetStore, build_file_attrs
from echo_core.verify import _open_readonly, verify


def generate(scene, limit=None, resume=False, force=False, flush_every=36):
    log = setup_logging(scene)
    _install_signal_handlers(log)

    loc_ids, positions = load_scene_locations(scene)
    if limit is not None:
        loc_ids = loc_ids[:limit]
        log.warning("--limit %d: generuje tylko %d z %d lokalizacji. Plik pozostanie NIEKOMPLETNY "
                    "(rozmiar ustalony na pelna scene), --verify zglosi brak probek.",
                    limit, len(loc_ids), len(positions))

    all_loc_ids = sorted(positions)  # rozmiar pliku zawsze na pelna scene
    attrs = build_file_attrs(scene, all_loc_ids)

    path = scene_h5(scene)
    if path.exists() and not (resume or force):
        raise SystemExit(
            f"{path} juz istnieje. Uzyj --resume, zeby dopisac brakujace probki, "
            f"albo --force, zeby wygenerowac od zera (NADPISZE istniejace dane).")
    mode = "r+" if (path.exists() and resume) else "w"
    if mode == "w" and path.exists():
        log.warning("--force: nadpisuje %s", path)

    store = DatasetStore(path, scene, all_loc_ids, positions, attrs=attrs, mode=mode,
                         flush_every=flush_every)
    log.info("Plik: %s (tryb %s), zapisanych probek na starcie: %d/%d",
             path, mode, store.n_written(), store.n_samples)
    if getattr(store, "_path_mismatch", None):
        old, new = store._path_mismatch
        log.warning("Plik zaczeto przy audio_sims_per_render=%d, dopisujemy przy %d. "
                    "Rownowaznosc obu sciezek potwierdzona pomiarem (GENERATOR_PARAMS.md §4.3), "
                    "ale scena bedzie mieszana — rozwaz regeneracje od zera (--force).", old, new)
    # Od razu, zeby --status w drugim terminalu widzial scene jako "W TOKU"
    # jeszcze zanim skonczy sie pierwsza lokalizacja.
    store.write_progress_sidecar({"running": True})

    t_start = time.time()
    renderer = None
    n_locations_done = 0
    n_list = []
    extra_renders_total = 0
    exit_code = 0
    interrupted = False

    decisions_fh = open(scene_decisions(scene), "a", encoding="utf-8")
    try:
        todo = [l for l in loc_ids if not store.location_done(l)]
        log.info("Do zrobienia: %d lokalizacji (%d juz kompletnych)",
                 len(todo), len(loc_ids) - len(todo))
        if not todo:
            log.info("Nic do zrobienia — wszystkie zadane lokalizacje sa kompletne.")
        else:
            renderer = Renderer(scene, log)
            # Rozgrzewka PRZED pierwsza lokalizacja: inaczej cala 8-renderowa
            # sonda pierwszej lokalizacji wypadlaby w okresie podwyzszonego szumu
            # i zawyzylaby N tej jednej lokalizacji o ~20-45 %.
            renderer.warmup(positions[todo[0]])

        for pos_in_todo, loc_id in enumerate(todo, start=1):
            if is_interrupted():
                interrupted = True
                break
            t_loc = time.time()
            position = positions[loc_id]
            missing = store.missing_angles(loc_id)

            # --- decyzja o N: sonda albo odtworzenie z pliku ------------------
            decision = store.get_decision(loc_id) if resume else None
            probe_specs = None
            probe_seconds = 0.0
            if decision is None:
                t_probe = time.time()
                probe_specs = [renderer.render(position, 0.0)[0] for _ in range(N_PROBE)]
                probe_seconds = time.time() - t_probe
                sigma_1, _h = sigma_1_from_specs(probe_specs)
                n_raw, n_planned, clamped = plan_n(sigma_1)
                store.put_decision(loc_id, sigma_1, n_raw, n_planned, clamped, probe_seconds)
            else:
                sigma_1 = decision["sigma_1_probe"]
                n_raw, n_planned, clamped = (decision["n_raw"], decision["n_planned"],
                                             decision["clamped"])
                log.info("  lok %d: decyzja odtworzona z pliku (N=%d) — sonda nie jest powtarzana",
                         loc_id, n_planned)

            # --- 36 orientacji ------------------------------------------------
            loc_extra = 0
            for angle in ANGLES_DEG:
                if angle not in missing:
                    continue

                if angle == 0 and probe_specs is not None:
                    # KRYTYCZNE — jednorodnosc szumu miedzy orientacjami.
                    #
                    # Przy N < 8 sonda daje wiecej renderow niz potrzeba dla
                    # orientacji 0 stopni. Nadmiarowe ODRZUCAMY — kazda z 36
                    # orientacji uzywa dokladnie N renderow.
                    #
                    # Powod nie jest kosztowy (strata to 0.06 h na caly zbior),
                    # tylko metodologiczny: przy N=6 i uzyciu wszystkich 8
                    # renderow sondy orientacja 0 stopni mialaby szum nizszy
                    # o sqrt(6/8) = 13 %. W zmierzonym rozkladzie N dotyczy to
                    # ~1/3 lokalizacji. A 0 stopni jest jedna z czterech
                    # orientacji bazowych: w warunku 4-kierunkowym to 1 z 4
                    # probek (25 %), w 36-kierunkowym 1 z 36 (2.8 %) — czyli
                    # SREDNI POZIOM SZUMU roznilby sie systematycznie miedzy
                    # warunkami ablacji, skorelowany dokladnie ze zmienna
                    # eksperymentalna. Darmowe do unikniecia, wiec unikamy.
                    specs = list(probe_specs[:n_planned])
                    n_probe_used = len(specs)
                    first_rgb = first_depth = None
                    while len(specs) < n_planned:
                        s, r, d = renderer.render(position, angle)
                        specs.append(s)
                        if first_rgb is None:
                            first_rgb, first_depth = r, d
                    if first_rgb is None:
                        # Wszystkie N renderow pochodzi z sondy, ktora zapisywala
                        # tylko spektrogramy — dorenderowujemy sam obraz.
                        # Rendering wizualny jest deterministyczny (PKL_FORMAT.md),
                        # wiec ten render jest identyczny z kazdym innym przy tej
                        # samej pozie; kosztuje 0.2 ms (audio dominuje).
                        _s, first_rgb, first_depth = renderer.render(position, angle)
                else:
                    specs = []
                    n_probe_used = 0
                    first_rgb = first_depth = None
                    for _ in range(n_planned):
                        s, r, d = renderer.render(position, angle)
                        specs.append(s)
                        if first_rgb is None:
                            first_rgb, first_depth = r, d

                # --- weryfikacja po fakcie (§3.4) ----------------------------
                snr_probe, _ = snr_from_specs(specs)
                snr_final = snr_probe
                extra = 0
                guard = 0
                while snr_final < TARGET_SNR and len(specs) < N_MAX:
                    guard += 1
                    if guard > 8:
                        raise RuntimeError(
                            f"petla weryfikacyjna nie zbiegla po 8 iteracjach "
                            f"(lok {loc_id}, kat {angle}, n={len(specs)}, snr={snr_final:.3f})")
                    # Nie dokladamy po jednym renderze: przeliczamy WYMAGANE n
                    # z aktualnego (dokladniejszego niz 8-renderowa sonda)
                    # oszacowania sigma_1 i skaczemy tam od razu. Dokladanie po
                    # jednym i sprawdzanie po kazdym byloby optional stopping —
                    # zatrzymywaloby sie dokladnie wtedy, gdy oszacowanie szumu
                    # akurat wypadnie nisko, czyli z obciazeniem w dol.
                    sigma_now, _ = sigma_1_from_specs(specs)
                    need = int(np.ceil((TARGET_SNR * sigma_now / SIGNAL_10DEG) ** 2))
                    need = int(min(max(need, len(specs) + 1), N_MAX))
                    for _ in range(need - len(specs)):
                        specs.append(renderer.render(position, angle)[0])
                        extra += 1
                    snr_final, _ = snr_from_specs(specs)

                loc_extra += extra
                extra_renders_total += extra

                echo = _mean_f32(specs)   # estymata = mean(|STFT|), domena "mag" (§1, §3.3 pkt 3)
                store.put_sample(
                    loc_id, angle, echo, first_rgb, first_depth, position,
                    {"snr_probe": snr_probe, "snr_final": snr_final,
                     "sigma_1_probe": sigma_1, "n_raw": n_raw, "n_planned": n_planned,
                     "n_total": len(specs), "n_rendered_extra": extra,
                     "n_probe": n_probe_used, "clamped": clamped})
                if store.maybe_flush():
                    store.write_progress_sidecar()
                else:
                    store.sidecar_heartbeat()

                if is_interrupted():
                    interrupted = True
                    break

            store.put_location_time(loc_id, time.time() - t_loc)
            store.maybe_flush(force=True)
            store.write_progress_sidecar()
            # Lokalizacja przerwana sygnalem ma zapisane tylko czesc katow — nie
            # liczy sie do sredniego N ani do tempa, bo zaklamalaby oba (jej czas
            # jest urwany w losowym miejscu).
            loc_complete = store.location_done(loc_id)
            if loc_complete:
                n_list.append(n_planned)
                n_locations_done += 1

            decisions_fh.write(json.dumps({
                "scene": scene, "loc_id": int(loc_id),
                "sigma_1_probe": round(float(sigma_1), 6),
                "n_raw": int(n_raw), "n_planned": int(n_planned), "clamped": clamped,
                "n_rendered_extra": int(loc_extra),
                "seconds": round(time.time() - t_loc, 2),
                "probe_seconds": round(probe_seconds, 2),
                "complete": bool(loc_complete),
                "utc": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            decisions_fh.flush()

            if loc_complete:
                per_loc = (time.time() - t_start) / n_locations_done
                eta = _fmt_hms(per_loc * (len(todo) - pos_in_todo))
                log.info("lok %3d/%-3d id=%-4d sigma1=%.5f N_raw=%-3d N=%-3d%s +%-3d dorend. | "
                         "sr.N %.1f | %5.1f s/lok | ETA %s",
                         pos_in_todo, len(todo), loc_id, sigma_1, n_raw, n_planned,
                         f" [{clamped}]" if clamped else "      ", loc_extra,
                         float(np.mean(n_list)), time.time() - t_loc, eta)
            else:
                log.warning("lok %3d/%-3d id=%-4d N=%-3d PRZERWANA — zapisano %d z %d katow, "
                            "decyzja o N zachowana w pliku; --resume dokonczy reszte",
                            pos_in_todo, len(todo), loc_id, n_planned,
                            N_ANGLES - len(store.missing_angles(loc_id)), N_ANGLES)

            if interrupted or is_interrupted():
                interrupted = True
                break

    except KeyboardInterrupt:
        log.error("Przerwane drugim sygnalem — plik moze zawierac niedokonczona probke "
                  "(flaga `written` chroni przed jej odczytem; --resume ja nadpisze)")
        exit_code = 130
    except BaseException:
        log.error("BLAD generacji — pelny traceback ponizej. Plik zostanie zamkniety, "
                  "dotychczasowe probki sa zapisane; wznowienie: --resume")
        log.error("%s", traceback.format_exc())
        exit_code = 1
    finally:
        decisions_fh.close()
        seconds_render = renderer.render_seconds if renderer else 0.0
        n_renders = renderer.n_renders if renderer else 0
        if renderer is not None:
            renderer.close()

        prev_sec = float(store.f.attrs.get("render_seconds_total", 0.0))
        prev_ren = int(store.f.attrs.get("renders_total", 0))
        runs = json.loads(store.f.attrs.get("runs", "[]"))
        runs.append({
            "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(),
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.time() - t_start, 1),
            "renders": n_renders,
            "warmup_renders": (renderer.n_warmup if renderer else 0),
            "locations": n_locations_done,
            "interrupted": bool(interrupted),
            "limit": limit,
            "hostname": socket.gethostname(),
            "exit_code": exit_code,
        })
        s_per_render = ((prev_sec + seconds_render) / (prev_ren + n_renders)
                        if (prev_ren + n_renders) else 0.0)
        store.close({
            "runs": json.dumps(runs),
            "render_seconds_total": prev_sec + seconds_render,
            "renders_total": prev_ren + n_renders,
            "seconds_per_render": s_per_render,
            "last_update_utc": datetime.now(timezone.utc).isoformat(),
            "complete": bool(store.n_written() == store.n_samples),
        })
        store.write_progress_sidecar({"complete": bool(store.n_written() == store.n_samples),
                                      "seconds_per_render": s_per_render,
                                      "interrupted": bool(interrupted)})

        wall = time.time() - t_start
        log.info("=" * 78)
        log.info("scena %s | lokalizacji w tym przebiegu %d | renderow %d | %.4f s/render",
                 scene, n_locations_done, n_renders, s_per_render)
        log.info("probek w pliku %d/%d (%.1f %%) | dorenderowanych renderow %d | czas %s",
                 store.n_written(), store.n_samples,
                 100.0 * store.n_written() / max(store.n_samples, 1), extra_renders_total,
                 _fmt_hms(wall))
        if n_list:
            log.info("N wybrane: mediana %d, srednia %.2f, zakres %d-%d",
                     int(np.median(n_list)), float(np.mean(n_list)), min(n_list), max(n_list))
        if interrupted:
            log.info("PRZERWANE czysto na granicy probki. Wznowienie:")
            log.info("  python %s --scene %s --resume", SCRIPT_PATH.name, scene)
        log.info("=" * 78)

    # Czyste przerwanie to nie blad — kod 0, zgodnie z wymaganiem odpornosci.
    return exit_code


# ---------------------------------------------------------------------------
# --probe-only: census sondy dla wszystkich lokalizacji sceny
# ---------------------------------------------------------------------------
def probe_census(scene, limit=None, force=False):
    """Sonda 8 renderow dla KAZDEJ lokalizacji sceny, bez 35 pozostalych orientacji.

    Po co: rozklad `N` znamy dotad z 52 pozycji (2-3 na scene). Sonda jest i tak
    pierwszym krokiem produkcji, wiec da sie ja wykonac osobno i tanio — 8
    renderow na lokalizacje zamiast ~36*N. Dla 1740 lokalizacji to ~14 tys.
    renderow (~35 min) zamiast ~600 tys. (~29 h). Daje PELNY rozklad `N_raw`
    zamiast ekstrapolacji, a wiec twarda odpowiedz, czy N_MAX pokrywa zbior.

    Logika sondy jest DOKLADNIE produkcyjna — te same funkcje (`Renderer`,
    `sigma_1_from_specs`, `plan_n`), ta sama rozgrzewka, ta sama kolejnosc
    wywolan. Inaczej pomiar nie mowilby nic o tym, co zrobi generator.

    `N_raw` zapisujemy SUROWE, przed clampem — o to wlasnie chodzi w tym pomiarze.
    """
    log = setup_logging(log_path=probe_census_log(scene))
    _install_signal_handlers(log)

    loc_ids, positions = load_scene_locations(scene)
    if limit is not None:
        loc_ids = loc_ids[:limit]

    csv_path = probe_census_csv(scene)
    done = {}
    if csv_path.exists() and not force:
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                done[int(row["loc_id"])] = row
        log.info("Wznowienie: %d lokalizacji juz w %s", len(done), csv_path.name)

    todo = [l for l in loc_ids if l not in done]
    log.info("%s: %d lokalizacji do sondowania (%d juz gotowych)",
             scene, len(todo), len(loc_ids) - len(todo))
    if not todo:
        log.info("Nic do zrobienia.")
        return 0

    PROBE_CENSUS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or force
    fh = open(csv_path, "w" if force else "a", newline="", encoding="utf-8")
    import csv as _csv
    writer = _csv.writer(fh)
    if write_header:
        writer.writerow(["scene", "loc_id", "sigma_1", "n_raw", "n_planned", "clamped",
                         "x", "y", "z", "probe_seconds"])
        fh.flush()

    renderer = None
    exit_code = 0
    t_start = time.time()
    n_done = 0
    try:
        renderer = Renderer(scene, log)
        renderer.warmup(positions[todo[0]])
        for i, loc_id in enumerate(todo, start=1):
            if is_interrupted():
                log.info("Przerwane po %d lokalizacjach — wznowienie: --probe-only --scene %s",
                         n_done, scene)
                break
            pos = positions[loc_id]
            t0 = time.time()
            specs = [renderer.render(pos, 0.0)[0] for _ in range(N_PROBE)]
            sigma_1, _h = sigma_1_from_specs(specs)
            n_raw, n_planned, clamped = plan_n(sigma_1)
            dt = time.time() - t0
            writer.writerow([scene, int(loc_id), f"{sigma_1:.6f}", int(n_raw),
                             int(n_planned), clamped, f"{pos[0]:.6f}", f"{pos[1]:.6f}",
                             f"{pos[2]:.6f}", f"{dt:.2f}"])
            fh.flush()
            n_done += 1
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(todo) - i)
                log.info("  %4d/%-4d id=%-4d sigma1=%.5f N_raw=%-3d | %.2f s/lok | ETA %s",
                         i, len(todo), loc_id, sigma_1, n_raw, elapsed / n_done,
                         _fmt_hms(eta))
    except KeyboardInterrupt:
        log.error("Przerwane drugim sygnalem")
        exit_code = 130
    except BaseException:
        log.error("BLAD census — traceback ponizej")
        log.error("%s", traceback.format_exc())
        exit_code = 1
    finally:
        fh.close()
        if renderer is not None:
            log.info("scena %s | lokalizacji %d | renderow %d (w tym %d rozgrzewki) | "
                     "%.4f s/render | czas %s", scene, n_done, renderer.n_renders,
                     renderer.n_warmup,
                     renderer.render_seconds / max(renderer.n_renders, 1),
                     _fmt_hms(time.time() - t_start))
            renderer.close()
    return exit_code


# ---------------------------------------------------------------------------
# --dry-run (bez GPU)
# ---------------------------------------------------------------------------
def dry_run(scene, limit=None):
    log = setup_logging()
    loc_ids, positions = load_scene_locations(scene)
    n_all = len(loc_ids)
    if limit is not None:
        loc_ids = loc_ids[:limit]
    n_loc = len(loc_ids)
    n_samples = n_loc * N_ANGLES

    # Renderow na lokalizacje: sonda (N_PROBE) + 36*N, minus te rendery sondy,
    # ktore zostana wykorzystane przy 0 stopni (min(N_PROBE, N)).
    n_mean = MEAN_N_SPEC
    renders_per_loc = N_PROBE + N_ANGLES * n_mean - min(N_PROBE, n_mean)
    renders = n_loc * renders_per_loc
    seconds = renders * S_PER_RENDER_SPEC
    raw_bytes = n_samples * BYTES_PER_SAMPLE

    y_values = np.array([positions[l][1] for l in loc_ids])

    print(f"\n=== --dry-run: {scene} ===")
    print(f"  plik wyjsciowy            {scene_h5(scene)}")
    print(f"  istnieje                  {'TAK' if scene_h5(scene).exists() else 'nie'}")
    print(f"  scena (mesh)              {scene_mesh(scene)}")
    print(f"    istnieje                {'TAK' if scene_mesh(scene).exists() else 'NIE — BLAD'}")
    print(f"  navmesh                   {'TAK' if (SCENE_ROOT / scene / 'habitat/mesh_semantic.navmesh').exists() else 'NIE — BLAD'}")
    print(f"  lokalizacje (pkl)         {n_loc}" + (f" z {n_all} (--limit)" if limit else ""))
    print(f"  orientacje                {N_ANGLES} (co 10 st.)")
    print(f"  PROBKI                    {n_samples}")
    print(f"  y agenta (z graph.pkl)    {y_values[0]:.6f}"
          f"{'' if np.ptp(y_values) == 0 else f' (UWAGA: niestale, rozrzut {np.ptp(y_values):.6f})'}")
    print(f"  pozycja pierwsza          {positions[loc_ids[0]]}")
    print()
    print(f"  zakladane srednie N       {n_mean} (GENERATOR_PARAMS.md §3.1)")
    print(f"  renderow / lokalizacje    {renders_per_loc:.1f}  (sonda {N_PROBE} + 36xN - odzysk)")
    print(f"  RENDEROW LACZNIE          {renders:,.0f}".replace(",", " "))
    print(f"  tempo (spec)              {S_PER_RENDER_SPEC} s/render")
    print(f"  CZAS SZACOWANY            {_fmt_hms(seconds)}  ({seconds/3600:.2f} h)")
    print(f"    + dorenderowanie §3.4   nieliczne, nieuwzglednione")
    print()
    print(f"  bajtow / probke           {BYTES_PER_SAMPLE:,}".replace(",", " ") +
          "  (echo 170 648 + rgb 65 536 + depth 65 536)")
    print(f"  ROZMIAR bez kompresji     {raw_bytes/2**30:.2f} GiB")
    print(f"    po gzip -4              mniej; rzeczywisty rozmiar w --status")
    # przez modul, nie przez stala z importu — patrz komentarz w echo_ctl.py
    from echo_core import paths as _P
    free = os.statvfs(_P.OUT_ROOT.parent if _P.OUT_ROOT.parent.exists() else REPO_ROOT)
    print(f"  wolne na dysku            {free.f_bavail * free.f_frsize / 2**30:.1f} GiB")
    print()
    print("  wejscia:")
    for label, p in (("chirp", CHIRP_PATH), ("materialy", MATERIAL_CONFIG),
                     ("lokalizacje", LOCATIONS_PKL), ("points.txt", points_txt(scene)),
                     ("graph.pkl", graph_pkl(scene))):
        print(f"    {label:<12} {'OK ' if p.exists() else 'BRAK'} {p}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", help="nazwa sceny Replica do wygenerowania (jedna, w jednym procesie)")
    p.add_argument("--limit", type=int, default=None,
                   help="tylko pierwsze N lokalizacji (smoke test); plik pozostanie niekompletny")
    p.add_argument("--resume", action="store_true",
                   help="dopisz brakujace probki do istniejacego pliku")
    p.add_argument("--force", action="store_true",
                   help="nadpisz istniejacy plik od zera (NISZCZY dane)")
    p.add_argument("--dry-run", action="store_true",
                   help="bez GPU: liczba lokalizacji, probek, szacowany czas i rozmiar")
    p.add_argument("--probe-only", action="store_true",
                   help="tylko sonda 8 renderow na KAZDA lokalizacje sceny (bez 35 pozostalych "
                        "orientacji) -> census sigma_1 i N_raw do CSV w outputs/probe_census/")
    p.add_argument("--verify", metavar="SCENA", help="pelna walidacja gotowego pliku (bez GPU)")
    p.add_argument("--status", action="store_true",
                   help="tabelka wszystkich 18 scen (bez GPU)")
    p.add_argument("--flush-every", type=int, default=36,
                   help="flush pliku HDF5 co N probek (domyslnie 36 = jedna lokalizacja)")
    p.add_argument("--plots", type=int, default=3,
                   help="ile losowych probek zapisac jako PNG w --verify (0 = zadnych)")
    p.add_argument("--variant", default="main", choices=list(VARIANTS),
                   help="main = geometria oryginalna (wariant glowny); "
                        "patched = sceny z domknietymi dziurami (wariant dodatkowy). "
                        "Kazdy wariant ma WLASNY katalog wyjsciowy, wiec sie nie mieszaja.")
    args = p.parse_args()

    # PRZED czymkolwiek, co dotyka sciezek albo listy scen: wariant przestawia
    # OUT_ROOT i scene_mesh() w echo_core.paths.
    set_variant(args.variant)
    scenes = scenes_for_variant()

    modes = [bool(args.status), bool(args.verify), bool(args.dry_run), bool(args.scene),
             bool(args.probe_only)]
    if sum(modes) == 0:
        p.print_help()
        return 2
    if args.status:
        return status()
    if args.verify:
        if args.verify not in scenes:
            raise SystemExit(f"nieznana scena: {args.verify}\n"
                             f"dostepne w wariancie {args.variant}: {', '.join(scenes)}")
        return verify(args.verify, n_plots=args.plots)
    if not args.scene:
        raise SystemExit("--dry-run wymaga --scene")
    if args.scene not in scenes:
        extra = ""
        if args.variant != "main" and args.scene in SCENE_ORDER:
            extra = ("\n  Ta scena jest SZCZELNA — nie ma laty, wiec w wariancie 'patched' jej"
                     "\n  geometria bylaby identyczna jak w 'main'. Uzyj wariantu glownego.")
        raise SystemExit(f"nieznana scena: {args.scene}\n"
                         f"dostepne w wariancie {args.variant}: {', '.join(scenes)}{extra}")
    if args.dry_run:
        return dry_run(args.scene, limit=args.limit)
    if args.probe_only:
        return probe_census(args.scene, limit=args.limit, force=args.force)
    return generate(args.scene, limit=args.limit, resume=args.resume, force=args.force,
                    flush_every=args.flush_every)


if __name__ == "__main__":
    sys.exit(main())
