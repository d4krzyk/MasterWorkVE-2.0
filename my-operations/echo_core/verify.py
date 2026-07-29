"""--verify: pelna walidacja gotowego pliku sceny. Bez GPU."""

import numpy as np

from .params import (ANGLES_DEG, BYTES_PER_SAMPLE, N_ANGLES, N_MAX, N_PROBE, S_PER_RENDER_SPEC,
                     TARGET_SNR)
from .paths import scene_h5, scene_verify_dir
from .scenes import load_scene_locations
from .store import SPEC_SHAPE

def _open_readonly(path):
    import h5py
    # locking=False pozwala czytac plik, ktory inny proces trzyma otwarty do
    # zapisu — dzieki temu --verify/--status dzialaja w drugim terminalu.
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _hist(values, bins=None, width=46):
    values = np.asarray(values)
    if values.size == 0:
        return ["    (brak danych)"]
    if bins is None:
        uniq = np.unique(values)
        if uniq.size <= 20 and np.all(uniq == uniq.astype(int)):
            counts = [(int(u), int((values == u).sum())) for u in uniq]
            top = max(c for _, c in counts)
            return [f"    {u:>6} | {'#' * max(1, int(width * c / top)):<{width}} {c}"
                    for u, c in counts]
        bins = 12
    counts, edges = np.histogram(values, bins=bins)
    top = max(counts.max(), 1)
    return [f"    {edges[i]:>7.3f}-{edges[i+1]:<7.3f} | "
            f"{'#' * max(0, int(width * counts[i] / top)):<{width}} {counts[i]}"
            for i in range(len(counts))]


def verify(scene, n_plots=3, seed=0):
    print(f"\n{'=' * 78}\n  WERYFIKACJA: {scene}\n{'=' * 78}")
    path = scene_h5(scene)
    if not path.exists():
        print(f"\n  WERDYKT: FAIL — plik nie istnieje: {path}")
        return 1

    failures, warnings_ = [], []
    f = _open_readonly(path)
    try:
        written = f["written"][:].astype(bool)
        n_slots = written.size
        n_written = int(written.sum())
        idx = np.flatnonzero(written)

        loc_id = f["location_id"][:][idx]
        angle = f["angle_deg"][:][idx]
        n_planned = f["n_planned"][:][idx].astype(int)
        n_total = f["n_total"][:][idx].astype(int)
        n_extra = f["n_rendered_extra"][:][idx].astype(int)
        n_raw = f["n_raw"][:][idx].astype(int)
        snr_probe = f["snr_probe"][:][idx].astype(float)
        snr_final = f["snr_final"][:][idx].astype(float)
        sigma_1 = f["sigma_1_probe"][:][idx].astype(float)
        clamped = np.array([c.decode() for c in f["clamped"][:][idx]])

        target_snr = float(f.attrs.get("target_snr", TARGET_SNR))
        n_max = int(f.attrs.get("n_max", N_MAX))
        n_loc_expected = int(f.attrs.get("n_locations", n_slots // N_ANGLES))
        expected_samples = n_loc_expected * N_ANGLES

        # ---------------- KOMPLETNOSC -----------------------------------
        print("\n--- KOMPLETNOSC ---")
        print(f"  probek zapisanych         {n_written} / {expected_samples} "
              f"({100.0*n_written/max(expected_samples,1):.2f} %)")
        if n_written != expected_samples:
            failures.append(f"kompletnosc: {n_written} probek zamiast {expected_samples} "
                            f"({n_loc_expected} lokalizacji x {N_ANGLES})")

        locs_present = np.unique(loc_id)
        print(f"  lokalizacji obecnych      {locs_present.size} / {n_loc_expected}")
        bad_angles = []
        for lid in locs_present:
            angs = angle[loc_id == lid]
            if angs.size != N_ANGLES or set(angs.tolist()) != set(ANGLES_DEG):
                bad_angles.append((int(lid), angs.size, len(set(angs.tolist()))))
        if bad_angles:
            failures.append(f"{len(bad_angles)} lokalizacji nie ma kompletu 36 unikalnych katow "
                            f"(np. {bad_angles[:5]})")
        else:
            print(f"  komplet 36 katow          OK dla wszystkich {locs_present.size} lokalizacji")
        if np.unique(np.stack([loc_id, angle]), axis=1).shape[1] != loc_id.size:
            failures.append("wystepuja zduplikowane pary (location_id, angle_deg)")
        else:
            print("  duplikaty (lok, kat)      brak")

        try:
            expected_locs = set(load_scene_locations(scene)[0])
            got = set(int(x) for x in locs_present)
            if got - expected_locs:
                failures.append(f"location_id spoza scene_observations_128.pkl: "
                                f"{sorted(got - expected_locs)[:10]}")
            missing_locs = expected_locs - got
            if missing_locs:
                failures.append(f"brakuje {len(missing_locs)} lokalizacji z pkl: "
                                f"{sorted(missing_locs)[:10]}")
            else:
                print(f"  zgodnosc z pkl            OK ({len(expected_locs)} lokalizacji)")
        except Exception as e:   # brak pkl nie jest bledem DANYCH, tylko srodowiska
            warnings_.append(f"nie udalo sie sprawdzic zbioru lokalizacji wzgledem pkl: {e}")

        # ---------------- INTEGRALNOSC ----------------------------------
        print("\n--- INTEGRALNOSC ---")
        expected_dtypes = {"echo": "float16", "rgb": "uint8", "depth": "float32",
                           "location_id": "int32", "angle_deg": "int16",
                           "position": "float32", "snr_probe": "float32",
                           "snr_final": "float32", "n_total": "int16",
                           "n_rendered_extra": "int16", "n_planned": "int16",
                           "n_raw": "int32", "sigma_1_probe": "float32"}
        expected_shapes = {"echo": SPEC_SHAPE, "rgb": (128, 128, 4), "depth": (128, 128),
                           "position": (3,)}
        for name, dt in expected_dtypes.items():
            if str(f[name].dtype) != dt:
                failures.append(f"dtype {name}: {f[name].dtype}, oczekiwano {dt}")
        for name, sh in expected_shapes.items():
            if tuple(f[name].shape[1:]) != sh:
                failures.append(f"ksztalt {name}: {f[name].shape[1:]}, oczekiwano {sh}")
        if not failures:
            print(f"  ksztalty i dtype          OK (echo {f['echo'].shape}, "
                  f"rgb {f['rgb'].shape}, depth {f['depth'].shape})")

        # Skan w kawalkach — caly `echo` nie miesci sie wygodnie w RAM dla duzych scen.
        echo_max = 0.0
        n_nan = n_inf = n_zero = n_depth_bad = 0
        depth_min, depth_max = np.inf, -np.inf
        CH = 256
        for start in range(0, n_slots, CH):
            sl = slice(start, min(start + CH, n_slots))
            mask = written[sl]
            if not mask.any():
                continue
            e = f["echo"][sl][mask].astype(np.float32)
            d = f["depth"][sl][mask]
            n_nan += int(np.isnan(e).sum() + np.isnan(d).sum())
            n_inf += int(np.isinf(e).sum() + np.isinf(d).sum())
            n_zero += int((~e.any(axis=(1, 2, 3))).sum())
            echo_max = max(echo_max, float(e.max()))
            depth_min = min(depth_min, float(d.min()))
            depth_max = max(depth_max, float(d.max()))
            n_depth_bad += int((d < 0).sum())

        print(f"  NaN / Inf                 {n_nan} / {n_inf}")
        if n_nan or n_inf:
            failures.append(f"NaN={n_nan}, Inf={n_inf} w echo/depth")
        print(f"  probki z samych zer        {n_zero}")
        if n_zero:
            failures.append(f"{n_zero} probek `echo` zlozonych z samych zer")
        print(f"  echo max                  {echo_max:.4f}  (float16 max 65504, zapas "
              f"{65504/max(echo_max,1e-9):.0f}x)")
        if echo_max >= 65504:
            failures.append(f"echo max {echo_max} osiaga granice float16 — nastapilo obciecie")
        print(f"  depth zakres              {depth_min:.3f} - {depth_max:.3f} m")
        if n_depth_bad:
            failures.append(f"{n_depth_bad} pikseli depth < 0")
        if depth_max > 30.0:
            warnings_.append(f"depth max {depth_max:.2f} m > 30 m — nietypowe dla Repliki "
                             f"(referencja z PKL_FORMAT.md: 12.66 m)")

        # ---------------- POPRAWNOSC METODOLOGICZNA ---------------------
        print("\n--- POPRAWNOSC METODOLOGICZNA ---")
        # N_MAX jest twardym limitem CALKOWITEJ liczby renderow probki (§5 ogr. 6:
        # probki obciete moga nie osiagnac SNR 3.5 i gwarancja ich nie obejmuje),
        # wiec gwarancje sprawdzamy tam, gdzie limit NIE zadzialal.
        capped = n_total >= n_max
        guaranteed = ~capped
        below = guaranteed & (snr_final < target_snr)
        ok_all = int((snr_final >= target_snr).sum())
        print(f"  snr_final >= {target_snr}          {ok_all} / {len(snr_final)} probek "
              f"(min {snr_final.min():.3f})")
        if below.any():
            failures.append(f"{int(below.sum())} probek nieobcietych przez N_MAX ma "
                            f"snr_final < {target_snr} (min {snr_final[below].min():.3f})")

        # Probki, ktorym petla weryfikacyjna dobila do N_MAX. Gwarancja jakosci
        # ich nie obejmuje (§5 ograniczenie 6), wiec musza byc WYMIENIONE, a nie
        # tylko wylaczone z mianownika — inaczej raport milczaco gubi przypadki,
        # ktore maja trafic do pracy.
        if capped.any():
            cap_ok = capped & (snr_final >= target_snr)
            cap_bad = capped & (snr_final < target_snr)
            print(f"  przy limicie N_MAX={n_max}       {int(capped.sum())} probek "
                  f"({int(cap_ok.sum())} osiagnelo prog mimo limitu, "
                  f"{int(cap_bad.sum())} nie)")
            for i in np.flatnonzero(capped)[:20]:
                print(f"      lok {loc_id[i]:<5} kat {angle[i]:<4} n_planned={n_planned[i]:<3} "
                      f"+{n_extra[i]:<3} = {n_total[i]:<3} snr_probe={snr_probe[i]:.3f} "
                      f"snr_final={snr_final[i]:.3f}")
            if cap_bad.any():
                warnings_.append(
                    f"{int(cap_bad.sum())} probek dobilo do N_MAX={n_max} i NIE osiagnelo "
                    f"SNR {target_snr} — kategoria z §5 ograniczenie 6, do wypunktowania "
                    f"w pracy (liczba, udzial, rozklad snr_final i n_raw)")
        else:
            print(f"  przy limicie N_MAX={n_max}       brak")

        # Regula spojnosci §3.4.1: n_rendered_extra > 0 <=> snr_probe < TARGET_SNR.
        # Wyjatek strukturalny: probka, ktora juz na starcie ma n_planned == N_MAX,
        # nie MOZE dostac dodatkowych renderow — tam implikacja "w prawo" nie obowiazuje.
        at_cap_from_start = n_planned >= n_max
        chk = ~at_cap_from_start
        viol = ((n_extra > 0) != (snr_probe < target_snr)) & chk
        print(f"  regula (extra>0 <=> snr_probe<{target_snr})  "
              f"{'OK' if not viol.any() else f'{int(viol.sum())} NARUSZEN'} "
              f"({int(chk.sum())} sprawdzonych, {int(at_cap_from_start.sum())} pominietych na N_MAX)")
        if viol.any():
            k = np.flatnonzero(viol)[:5]
            failures.append(
                "naruszenie reguly spojnosci petli weryfikacyjnej dla "
                f"{int(viol.sum())} probek, np. " +
                "; ".join(f"lok {loc_id[i]} kat {angle[i]}: snr_probe={snr_probe[i]:.3f}, "
                          f"extra={n_extra[i]}" for i in k))

        bad_total = n_total != (n_planned + n_extra)
        print(f"  n_total == n_planned + extra  "
              f"{'OK' if not bad_total.any() else f'{int(bad_total.sum())} NARUSZEN'}")
        if bad_total.any():
            failures.append(f"{int(bad_total.sum())} probek ma n_total != n_planned + n_rendered_extra")

        nonuniform = []
        for lid in locs_present:
            vals = np.unique(n_planned[loc_id == lid])
            if vals.size != 1:
                nonuniform.append((int(lid), vals.tolist()))
        print(f"  n_planned jednolite w lokalizacji  "
              f"{'OK' if not nonuniform else f'{len(nonuniform)} NIEJEDNOLITYCH'}")
        if nonuniform:
            failures.append(f"n_planned rozni sie miedzy orientacjami w {len(nonuniform)} "
                            f"lokalizacjach: {nonuniform[:5]} — zlamana zasada z §3.3 pkt 2")

        for kind in ("min", "max"):
            sel = clamped == kind
            if not sel.any():
                print(f"  clamp '{kind}'              brak")
                continue
            locs = sorted(set(int(x) for x in loc_id[sel]))
            print(f"  clamp '{kind}'              {len(locs)} lokalizacji, "
                  f"{int(sel.sum())} probek")
            for lid in locs[:20]:
                m = sel & (loc_id == lid)
                print(f"      lok {lid:<5} n_raw={n_raw[m][0]:<4} n_planned={n_planned[m][0]:<3} "
                      f"snr_final {snr_final[m].min():.2f}-{snr_final[m].max():.2f}")
            if kind == "max":
                bad = sel & (snr_final < target_snr)
                if bad.any():
                    warnings_.append(
                        f"{int(bad.sum())} probek z clamped=='max' ma snr_final < {target_snr} "
                        f"— to kategoria z §5 ograniczenie 6, do wypunktowania w pracy")

        # ---------------- STATYSTYKI ------------------------------------
        print("\n--- STATYSTYKI DO PRACY ---")
        print(f"  histogram n_planned  (mediana {int(np.median(n_planned))}, "
              f"srednia {n_planned.mean():.2f})")
        for line in _hist(n_planned):
            print(line)
        print(f"\n  histogram snr_probe  (mediana {np.median(snr_probe):.3f}, "
              f"min {snr_probe.min():.3f}, max {snr_probe.max():.3f}, "
              f"ponizej progu {int((snr_probe < target_snr).sum())})")
        for line in _hist(snr_probe, bins=12):
            print(line)
        print(f"\n  histogram snr_final  (mediana {np.median(snr_final):.3f}, "
              f"min {snr_final.min():.3f}, max {snr_final.max():.3f})")
        for line in _hist(snr_final, bins=12):
            print(line)
        # sigma_1_probe jest stale w obrebie lokalizacji (decyzja zapada raz na
        # lokalizacje, §3.3), wiec histogram po probkach powielilby kazda wartosc
        # 36 razy — bierzemy po jednej na lokalizacje.
        sigma_per_loc = np.array([sigma_1[loc_id == lid][0] for lid in locs_present])
        print(f"\n  sigma_1_probe (na lokalizacje)  mediana {np.median(sigma_per_loc):.5f}, "
              f"zakres {sigma_per_loc.min():.5f}-{sigma_per_loc.max():.5f}")
        for line in _hist(sigma_per_loc, bins=10):
            print(line)
        print(f"\n  dorenderowane         {int((n_extra > 0).sum())} probek "
              f"({100.0*(n_extra > 0).mean():.2f} %), lacznie {int(n_extra.sum())} renderow")
        print(f"  renderow w pliku      {int(n_total.sum())} "
              f"(sonda liczona osobno: {locs_present.size} x {int(f.attrs.get('n_probe', N_PROBE))})")
        spr = float(f.attrs.get("seconds_per_render", 0.0))
        print(f"  tempo zmierzone       {spr:.4f} s/render "
              f"(spec {S_PER_RENDER_SPEC}, {100*(spr/S_PER_RENDER_SPEC - 1):+.1f} %)")
        print(f"  rozmiar pliku         {path.stat().st_size/2**20:.1f} MiB "
              f"({path.stat().st_size/max(n_written,1)/1024:.1f} KiB/probke, "
              f"bez kompresji {BYTES_PER_SAMPLE/1024:.1f})")

        # ---------------- INSPEKCJA WZROKOWA ----------------------------
        if n_plots > 0 and n_written > 0:
            out_dir = scene_verify_dir(scene)
            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            pick = rng.choice(idx, size=min(n_plots, idx.size), replace=False)
            made = _plot_samples(f, pick, out_dir, scene)
            print("\n--- INSPEKCJA WZROKOWA ---")
            for p in made:
                print(f"  {p}")
    finally:
        f.close()

    print(f"\n{'=' * 78}")
    if warnings_:
        print("  OSTRZEZENIA:")
        for w in warnings_:
            print(f"    - {w}")
    if failures:
        print(f"  WERDYKT: FAIL ({len(failures)})")
        for x in failures:
            print(f"    - {x}")
    else:
        print("  WERDYKT: PASS")
    print("=" * 78)
    return 1 if failures else 0


def _plot_samples(f, indices, out_dir, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for i in indices:
        echo = f["echo"][i].astype(np.float32)
        rgb = f["rgb"][i]
        depth = f["depth"][i]
        lid = int(f["location_id"][i])
        ang = int(f["angle_deg"][i])
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for ch in (0, 1):
            im = axes[ch].imshow(echo[ch], origin="lower", aspect="auto", cmap="magma")
            axes[ch].set_title(f"spektrogram, kanal {ch} ({'L' if ch == 0 else 'P'})")
            axes[ch].set_xlabel("ramka")
            axes[ch].set_ylabel("prazek czestotliwosci")
            fig.colorbar(im, ax=axes[ch], fraction=0.046)
        axes[2].imshow(rgb[..., :3])
        axes[2].set_title("RGB")
        axes[2].axis("off")
        im = axes[3].imshow(depth, cmap="viridis")
        axes[3].set_title("depth [m]")
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046)
        fig.suptitle(f"{scene}  lok={lid}  kat={ang} st.  "
                     f"N={int(f['n_total'][i])}  snr_final={float(f['snr_final'][i]):.2f}")
        fig.tight_layout()
        # Nazwa generyczna — plik nie opuszcza katalogu sceny, wiec nazwa sceny
        # w nazwie pliku bylaby redundancja (inaczej niz w przypadku .h5).
        p = out_dir / f"loc{lid}_ang{ang}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


