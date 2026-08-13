#!/usr/bin/env python
"""Eksport KAZDEJ zmierzonej liczby do pisania pracy. Zero GPU.

    python my-operations/ml/thesis_numbers.py

Produkuje dwa pliki:
    outputs/ml/thesis_numbers.json      -- zrodlo maszynowe
    my-operations/docs/LICZBY_DO_PRACY.md -- do czytania przy pisaniu

Kazda pozycja niesie: wartosc, jednostke, status [Z]/[Z-]/[W], plik dowodowy i sekcje raportu,
w ktorej jest omowiona. Liczby z LITERATURY sa w osobnej grupie i jawnie oznaczone -- silnik
akustyczny Gao i Paridy jest inny niz nasz, wiec porownanie jest wylacznie WEWNETRZNE (porzadek
warunkow i rzad wielkosci), nigdy bezposrednie.

Na koncu: lista liczb, KTORYCH JESZCZE NIE MA, z odsylaczem do warunku, ktory je da -- zeby autor
wiedzial, gdzie w tekscie zostawic luke, zamiast odkrywac brak przy skladaniu tabeli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402

R10 = "RAPORT_SESJI_2026-08-10.md"
R11 = "RAPORT_SESJI_2026-08-11.md"
R05 = "RAPORT_SESJI_2026-08-05.md"
R13 = "RAPORT_SESJI_2026-08-13.md"
GP = "docs/GENERATOR_PARAMS.md"

GRUPY = [
    ("dataset", "1. Zbiór danych"),
    ("silnik", "2. Charakterystyka silnika akustycznego"),
    ("geometria", "3. Geometria `main` vs `patched`"),
    ("determinizm", "4. Determinizm i wydajność"),
    ("wyniki_echo", "5. Wyniki grupy `echo`"),
    ("budzet", "6. Budżet obliczeniowy i dyskowy"),
    ("literatura", "7. Odniesienia z literatury — NIE nasze pomiary"),
]


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect() -> tuple[list[dict], list[dict]]:
    O = paths.ML_OUTPUTS
    gc = _load(O / "geometry_check" / "geometry_check.json") or {}
    det = _load(O / "determinism" / "determinism_check.json") or {}
    abl = _load(O / "echo_ablation" / "echo_ablation.json") or {}
    den = _load(O / "echo_ablation" / "echo_density_seed0.json") or {}
    gap = _load(O / "echo_ablation" / "gap_table_seed0.json") or {}
    exp = _load(O / "experiments.json") or {}
    spl = _load(O / "splits" / "replica_locations.json") or {}
    vl = _load(O / "verify_loader" / "main" / "verify_loader.json") or {}
    disk = _load(O / "disk_budget.json") or {}
    mask = _load(O / "mask_check" / "mask_check.json") or {}

    N = []
    def add(g, name, val, unit, status, evidence, sec, note=""):
        N.append({"grupa": g, "nazwa": name, "wartosc": val, "jednostka": unit,
                  "status": status, "dowod": evidence, "sekcja": sec, "uwaga": note})

    # ---- 1. dataset
    m = spl.get("meta", {})
    add("dataset", "Scen Replica łącznie", 18, "scen", "[Z]", "splits/replica_locations.json", f"{R05} §3.4")
    add("dataset", "Scen treningowych / held-out", "15 / 3", "scen", "[Z]",
        "splits/replica_locations.json", f"{R05} §3.2", "held-out: apartment_2, frl_apartment_5, office_4")
    nl = m.get("n_locations", {})
    add("dataset", "Lokalizacji train / val / test", f"{nl.get('train')} / {nl.get('val')} / {nl.get('test')}",
        "lokalizacji", "[Z]", "splits/replica_locations.json", f"{R05} §3.2")
    add("dataset", "Odcisk podziału", m.get("location_fingerprint"), "sha256[:16]", "[Z]",
        "splits/replica_locations.json", f"{R05} §3.2", "ZAMROŻONY — nie regenerować")
    add("dataset", "Orientacji na lokalizację", 36, "kątów co 10°", "[Z]", "atrybuty HDF5", f"{GP} §1")
    add("dataset", "Próbek łącznie, wariant main", 62640, "próbek", "[Z]",
        "verify_loader/main/verify_loader.json", f"{R05} §3.4")
    add("dataset", "Próbek łącznie, wariant patched", 44064, "próbek", "[Z]", f"{GP} §4.5", f"{GP} §4.5",
        "10 scen łatanych; do treningu + 8 scen szczelnych z main")
    add("dataset", "Kształt spektrogramu", "(2, 257, 166)", "kanały × f × t", "[Z]",
        "atrybuty HDF5", f"{R05} §3.1", "bit-zgodny z generate_spectrogram() Paridy")
    add("dataset", "max_depth (Replica)", 14.104, "m", "[Z]", "base_options.py Paridy", f"{R05} §3.5")
    add("dataset", "Pikseli powyżej max_depth", "131 (1,3·10⁻⁵ %)", "pikseli", "[Z]",
        "verify_loader/main/verify_loader.json", f"{R05} §3.5", "wszystkie w apartment_0 (scena treningowa)")
    add("dataset", "Udział pikseli depth == 0, main", 8.480338, "%", "[Z]",
        "verify_loader/main/verify_loader.json", f"{R10} §1.1")
    add("dataset", "Udział pikseli depth == 0, patched", 0.212069, "%", "[Z]",
        "verify_loader/patched/verify_loader.json", f"{R10} §1.1")
    add("dataset", "Udział pikseli krawędziowych (próg 0,10 m/px)", 11.3, "%", "[Z]",
        "eval/*/eval.json", f"{R05} §3.12", "0,05 m/px → 21,54 %; 0,20 m/px → 5,86 %")

    # ---- 2. silnik
    add("silnik", "indirectRayCount", 500, "promieni", "[Z]", "atrybuty HDF5", f"{GP} §1")
    add("silnik", "threadCount", 1, "wątek", "[Z]", "atrybuty HDF5", f"{GP} §1", "wątki dzielą budżet promieni")
    add("silnik", "Sygnał przy 10° (SIGNAL_10DEG)", 0.0644, "RMSE spektrogramu", "[Z]",
        "atrybuty HDF5", f"{GP} §3")
    add("silnik", "Docelowy SNR", 3.5, "—", "[Z]", "atrybuty HDF5", f"{GP} §3")
    add("silnik", "Szum render-do-renderu (ten sam kąt)", "0,03–0,16", "RMSE spektrogramu", "[Z]",
        "diagnose_rlr_noise_out/", "CLAUDE.md", "zależny od pozycji")
    add("silnik", "RMSE między renderami 90° od siebie", "0,30–0,35", "RMSE spektrogramu", "[Z]",
        "diagnose_rlr_noise_out/", "CLAUDE.md", "potwierdzone niezależnie w geometry_check: 0,3029–0,3159")
    add("silnik", "N adaptacyjne: min / max / próbne", "6 / 64 / 8", "renderów", "[Z]",
        "atrybuty HDF5", f"{GP} §3")
    add("silnik", "Czas renderu (materiały włączone)", "0,111–0,148", "s/render", "[Z]",
        "atrybuty HDF5", f"{GP} §4.3")
    add("silnik", "Przepustowość dataloadera (8 workerów)", 2645.1, "próbek/s", "[Z]",
        "bench/bench_main.json", f"{R05} §3.7")

    # ---- 3. geometria
    ch = gc.get("channels", {})
    if ch:
        f5 = ch.get("frl_apartment_5", {}).get("channels", {})
        add("geometria", "Zmienionych wartości depth (frl_apartment_5)", 17.571, "% wartości", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.1")
        add("geometria", "Zmienionych wartości echo (wszystkie sceny)", "91,9–93,2", "% komórek", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.1")
        add("geometria", "Pikseli usuniętych przez łatkę (+ → 0)", 0, "pikseli", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.2",
            "we WSZYSTKICH 10 scenach — dlatego maska przecięcia = maska main")
    g = gc.get("echo_groups", {})
    ovs, mvp = g.get("open_vs_sealed", {}), g.get("main_vs_patched", {})
    if ovs:
        add("geometria", "Energia całkowita: otwarte vs szczelne", ovs.get("energy_change_pct_open_vs_sealed"),
            "%", "[Z]", "geometry_check/geometry_check.json", f"{R10} §2.5")
        add("geometria", "Energia PÓŹNA (pogłos): otwarte vs szczelne",
            ovs.get("LATE_energy_change_pct_open_vs_sealed"), "%", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.5",
            "to jest właściwa liczba — całkowita jest zdominowana przez ścieżkę bezpośrednią")
        add("geometria", "Względny kontrast kątowy późny: otwarte / szczelne",
            ovs.get("LATE_relative_contrast_ratio_open_over_sealed"), "×", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.6")
    if mvp:
        add("geometria", "Energia PÓŹNA: patched vs main", mvp.get("LATE_energy_change_pct_patched_vs_main"),
            "%", "[Z]", "geometry_check/geometry_check.json", f"{R10} §2.5")
        add("geometria", "Względny kontrast kątowy późny: patched / main",
            mvp.get("LATE_relative_contrast_ratio_patched_over_main"), "×", "[Z]",
            "geometry_check/geometry_check.json", f"{R10} §2.6",
            "domknięcie sufitu OBNIŻA kontrast kątowy przy jednoczesnym wzroście SNR — kompromis")
    add("geometria", "Granica części późnej spektrogramu", "ramka 30 (10,9 ms)", "—", "[W]",
        "geometry_check.py::LATE_FRAME_START", f"{R10} §2.5",
        "za pierwszym odbiciem podłoga/sufit przy 1,25 m (7,3 ms)")
    if mask:
        fr = mask.get("frac_kadru", {})
        add("geometria", "Piksele zmienione, a ważne w obu wariantach",
            round(fr.get("roznica_zmienione_a_wazne", 0) * 100, 3), "% kadru", "[Z]",
            "mask_check/mask_check.json", f"{R11} §5")
        no = mask.get("narzut_wzgledny", {})
        add("geometria", "Narzut maski przecięcia w ewaluacji",
            round(no.get("przeciecie", 0) * 100), "%", "[Z]", "mask_check/mask_check.json", f"{R11} §5")

    # ---- 4. determinizm
    v = det.get("verdict", {})
    if v:
        add("determinizm", "Podłoga szumu frameworka (|ΔRMSE| po 2000 krokach)",
            v.get("abs_rmse_diff_floor_slow"), "RMSE", "[Z]",
            "determinism/determinism_check.json", f"{R10} §3.1",
            "dwa przebiegi TEGO SAMEGO kodu, to samo ziarno; zakres po krokach 0,0021–0,0096")
        add("determinizm", "Rozbieżność podstawienia BilinearEinsum (|ΔRMSE|)",
            v.get("abs_rmse_diff_substitution"), "RMSE", "[Z]",
            "determinism/determinism_check.json", f"{R10} §3.1", "10× PONIŻEJ podłogi")
        add("determinizm", "Rozbieżność wag: podłoga / podstawienie",
            f"{v.get('rel_l2_floor_slow'):.3e} / {v.get('rel_l2_substitution'):.3e}", "względna L2", "[Z]",
            "determinism/determinism_check.json", f"{R10} §3.1")
        add("determinizm", "Wagi startowe nn.Bilinear vs BilinearEinsum",
            "bit-identyczne", "—", "[Z]", "determinism/determinism_check.json", f"{R10} §3.1")
    add("determinizm", "Przyspieszenie --fast-bilinear (pełna pętla)", 16.15, "×", "[Z]",
        "determinism/determinism_check.json", f"{R10} §3.1",
        "1,5391 → 0,0953 s/krok; mikrobenchmark samego kroku dawał 19,5×")
    add("determinizm", "Parametry: pełny model", 316918781, "parametrów", "[Z]",
        "experiments.py::PARAM_COUNTS", f"{R10} §5.5",
        "rgbdepth 16 658 561 + audio 8 984 073 + attention 279 581 505 + material 11 694 642")
    add("determinizm", "Parametry: echo2depth", 8984073, "parametrów", "[Z]",
        "experiments.py::PARAM_COUNTS", f"{R10} §5.5")
    add("determinizm", "Parametry: Model 2 (pretekst)", 25733446, "parametrów", "[Z]",
        "pretext/model.py", f"{R10} §6.2", "w tym RGBDepthNet 16 658 561 do przeniesienia")
    add("determinizm", "Zgodność metryk z implementacją Paridy", 1.494e-06, "max |różnica|", "[Z]",
        "metrics.py::test_matches_parida()", f"{R05} §3.11")
    add("determinizm", "Zgodność tabeli per próbka z akumulatorem", 2.745e-08, "max |różnica|", "[Z]",
        "metrics.py::test_table_matches_accumulator()", f"{R10} §4", "17 kontroli")

    # ---- 5. wyniki echo (3 ziarna, nowy protokol walidacji -- ZASTEPUJA wersje z 1 ziarna)
    s3 = _load(O / "echo_ablation" / "echo_3seeds.json") or {}
    fg = _load(O / "echo_ablation" / "full_model_gate.json") or {}
    if s3:
        for c, v in s3.get("RMSE_test36", {}).items():
            add("wyniki_echo", f"RMSE test@36: {c}", f"{v['mean']:.5f} ± {v['sd']:.5f}", "RMSE", "[Z]",
                "echo_ablation/echo_3seeds.json", f"{R11} §3", "średnia ± sd po 3 ziarnach")
        for lab, v in s3.get("kontrasty", {}).items():
            add("wyniki_echo", f"Kontrast: {lab}", f"{v['mean']:.5f} ± {v['sd_seeds']:.5f}", "RMSE", "[Z]",
                "echo_ablation/echo_3seeds.json", f"{R11} §3",
                f"sd po ZIARNACH; istotne we wszystkich 3: {v['wszystkie_istotne']}; "
                f"CI po lokalizacjach (ziarno 0) {v['ci_per_seed'][0]}")
        ug = s3.get("udzial_gestosci_pct", {})
        add("wyniki_echo", "Udział gęstości kątowej w efekcie łącznym",
            f"{ug.get('mean'):.1f} ± {ug.get('sd'):.1f}", "%", "[Z]",
            "echo_ablation/echo_3seeds.json", f"{R11} §3", "gęstość 2,36× większa niż ilość danych")
        lk = s3.get("luka_katowa_EA", {})
        add("wyniki_echo", "Luka generalizacji kątowej EA (0° → 40°)",
            f"{lk.get('mean'):.5f} ± {lk.get('sd'):.5f}", "RMSE", "[Z]",
            "echo_ablation/echo_3seeds.json", f"{R11} §3",
            f"{lk.get('rel_pct_mean'):.2f} %, monotoniczna w 3/3 ziarnach; "
            f"stary protokół dawał +0,35540 — zawyżał o {abs(lk.get('zmiana', 0)):.5f}")
        add("wyniki_echo", "Krzywa RMSE(odległość kątowa), EA",
            {k: round(v["mean"], 5) for k, v in sorted(s3.get("krzywa_katowa_EA", {}).items(),
                                                       key=lambda x: float(x[0]))},
            "RMSE per kubełek", "[Z]", "echo_ablation/echo_3seeds.json", f"{R11} §3",
            "kubełki 0/10/20/30/40°; 45° nie występuje przy siatce co 10°")
        add("wyniki_echo", "Luka test@36 − test@4 per warunek",
            {c: f"{v['mean']:.5f}" for c, v in s3.get("luka_test36_minus_test4", {}).items()},
            "RMSE", "[Z]", "echo_ablation/echo_3seeds.json", f"{R11} §3",
            "lukę ma WYŁĄCZNIE warunek bez pokrycia kątowego")
        add("wyniki_echo", "Rozrzut po ziarnach: EA vs pozostałe",
            "0,01066 wobec 0,0024–0,0036", "RMSE (sd)", "[Z]",
            "echo_ablation/echo_3seeds.json", f"{R11} §3",
            "warunek o najrzadszym pokryciu kątowym jest 3–4× wrażliwszy na inicjalizację")
    if fg:
        add("wyniki_echo", "c_full — wkład echa w PEŁNYM modelu", fg.get("c_full"), "RMSE", "[Z-]",
            "echo_ablation/full_model_gate.json", f"{R11} §2",
            f"95 % CI {[round(x,5) for x in fg.get('c_full_CI', [])]}, ziarno 0; "
            f"26,4× mniej niż w echo2depth")
        add("wyniki_echo", "Względny wkład echa w pełnym modelu",
            round(fg.get("kontekst", {}).get("wklad_wzgledny_full_pct", 0), 1), "%", "[Z-]",
            "echo_ablation/full_model_gate.json", f"{R11} §2",
            "u Gao 7,5 % — zgodność rzędu wielkości potwierdza poprawność potoku, "
            "NIE jest zestawieniem wyników")
        add("wyniki_echo", "EA vs EB na test@4 (nowy protokół, 3 ziarna)",
            "0,01787 ± 0,01128", "RMSE", "[Z]", "eval/compare_EA_seed*_vs_EB_seed*_test4.json",
            f"{R11} §4.1", "91,4 % kary EA powstaje na kątach NIEWIDZIANYCH")
    if abl:
        w = abl.get("wynik", {})
        add("wyniki_echo", "Całkowity wkład echa (echo2depth, walidacja)",
            w.get("calkowity_wklad_echa_RMSE"), "RMSE", "[Z-]",
            "echo_ablation/echo_ablation.json", f"{R10} §3.3", "ziarno 0")
    # `echo_density_seed0.json` (1 ziarno, stary protokol) NIE wchodzi juz do zestawienia
    # -- zastapiony przez `echo_3seeds.json`. Zostaje na dysku jako slad historyczny.
    if gap:
        add("wyniki_echo", "Luka test@36 − test@4 per warunek",
            {r["warunek"]: round(r["luka"], 5) for r in gap.get("wiersze", [])},
            "RMSE", "[Z-]", "echo_ablation/gap_table_seed0.json", f"{R11} §4.2",
            "lukę ma WYŁĄCZNIE warunek bez pokrycia kątowego")
    add("wyniki_echo", "EA vs EB na test@4 (sparowane, te same 732 próbki)", 0.00150, "RMSE", "[Z-]",
        "eval/compare_EA_seed0_vs_EB_seed0_test4.json", f"{R11} §4.1",
        "95 % CI [−0,01325; +0,01731] — OBEJMUJE ZERO")

    # ---- 5a. Wyniki kolejki nocnej 2026-08-13
    fr = _load(O / "echo_ablation" / "final_results_2026-08-13.json") or {}
    if fr:
        kr = fr.get("krzywa_stalego_budzetu", {})
        pts = kr.get("punkty", {})
        if pts:
            add("wyniki_echo", "KRZYWA STAŁEGO BUDŻETU: RMSE w funkcji siatki K",
                {f"K={k}": f"{v['mean']:.5f}" for k, v in
                 sorted(pts.items(), key=lambda x: int(x[0]))},
                "RMSE", "[Z]", "echo_ablation/final_results_2026-08-13.json", f"{R13} §1",
                "echo2depth, 4 próbki/lokalizację (5 496) w KAŻDYM punkcie — liczność stała, "
                "zmienia się wyłącznie siatka; 3 ziarna")
            k4, k9, k36 = pts["4"]["mean"], pts["9"]["mean"], pts["36"]["mean"]
            add("wyniki_echo", "Nasycenie krzywej stałego budżetu",
                f"4→9: {k4-k9:.3f} · 9→36: {k9-k36:.3f}", "RMSE", "[Z]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §1",
                "przejście 4→9 daje 6,7× więcej niż 9→36 — punkt odcięcia ok. K = 9–12")
            c = kr.get("kontrasty", {}).get("K4_do_K36", {})
            if c:
                add("wyniki_echo", "Krzywa stałego budżetu: K=4 → K=36", c.get("delta_point"),
                    "RMSE", "[Z]", "echo_ablation/final_results_2026-08-13.json", f"{R13} §1",
                    f"95 % CI [{c.get('ci_low'):.5f}; {c.get('ci_high'):.5f}], bootstrap po lokalizacjach")
        gl = fr.get("glowne_pelny_model_1_ziarno", {})
        if gl.get("punkty"):
            add("wyniki_echo", "PEŁNY MODEL: RMSE test@36 (A / B / D)",
                {k: f"{v['mean']:.5f}" for k, v in gl["punkty"].items()}, "RMSE", "[Z-]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §2",
                "1 ziarno (degradacja 2026-08-11 §2) — bez oszacowania rozrzutu po ziarnach")
            add("wyniki_echo", "Pełny model: gęstość (D−A) / ilość danych (B−D)",
                f"{gl['D_minus_A']:+.5f} / {gl['B_minus_D']:+.5f}", "RMSE", "[Z-]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §2",
                "n=1 ziarno; oba porównywalne z podłogą szumu 0,0023–0,0073 — patrz zastrzeżenie")
        ge = fr.get("geometria_echo2depth", {})
        if ge.get("patched_minus_main"):
            add("geometria", "Wpływ domknięcia geometrii (echo2depth, patched − main)",
                {k: f"{v:+.5f}" for k, v in ge["patched_minus_main"].items()}, "RMSE", "[Z-]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §3",
                "1 ziarno; wartości DODATNIE = `patched` GORSZY mimo +46 % energii pogłosu")
        m2 = fr.get("model2_pretekst", {})
        if m2.get("MAAE"):
            add("wyniki_echo", "Model 2: MAAE zadania pretekstowego",
                {k: f"{v:.2f}" for k, v in m2["MAAE"].items()}, "stopnie", "[Z]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §4",
                "poziom losowy 90° NIEZALEŻNIE od K")
            r = m2["rozklad"]
            add("wyniki_echo", "Model 2: rozkład efektu pretreningu",
                f"ilość par {r['ilosc_par_K36_minus_K36p16']:+.2f}° · "
                f"rozdzielczość {r['rozdzielczosc_K36p16_minus_K4']:+.2f}°", "stopnie", "[Z]",
                "echo_ablation/final_results_2026-08-13.json", f"{R13} §4",
                "CAŁA przewaga K=36 pochodzi z 81× większej liczby par, NIE z rozdzielczości kątowej")
        tr = fr.get("model2_transfer", {}).get("punkty", {})
        if tr:
            add("wyniki_echo", "Model 2: transfer RGB2Depth (5 ziaren)",
                {k: f"{v['mean']:.5f}" for k, v in sorted(tr.items(), key=lambda x: x[1]["mean"])},
                "RMSE", "[Z]", "echo_ablation/final_results_2026-08-13.json", f"{R13} §5",
                "WYNIK NEGATYWNY — żadna różnica wobec `scratch` nie jest istotna")
            for k, v in tr.items():
                if v.get("p") is not None:
                    add("wyniki_echo", f"Model 2: transfer {k} vs scratch",
                        f"{v['delta']:+.5f} (p={v['p']:.3f})", "RMSE", "[Z]",
                        "echo_ablation/final_results_2026-08-13.json", f"{R13} §5",
                        "test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch")

    # ---- 5b. Model 2 (pretekst + transfer). Puste, dopoki kolejka nie policzy.
    pm = _load(O / "pretext" / "summary.json") or {}
    if pm.get("pretext_by_K"):
        for k, v in sorted(pm["pretext_by_K"].items(), key=lambda x: int(float(x[0]))):
            add("wyniki_echo", f"Model 2: MAAE zadania pretekstowego, K={k}",
                f"{v['mean']:.2f}" + (f" ± {v['sd']:.2f}" if v.get("sd") else ""), "stopnie", "[Z]",
                "pretext/summary.json", "Model 2",
                "poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1")
    if pm.get("transfer_by_label"):
        for lab, v in sorted(pm["transfer_by_label"].items(), key=lambda x: x[1]["mean"]):
            add("wyniki_echo", f"Model 2: RGB2Depth po pretreningu — {lab}",
                f"{v['mean']:.5f}" + (f" ± {v['sd']:.5f}" if v.get("sd") else ""), "RMSE", "[Z]",
                "pretext/summary.json", "Model 2",
                f"n_ziaren={v['n_seeds']}; zadanie docelowe BEZ audio w czasie testu")
    if pm.get("rozklad_efektu_pretreningu"):
        r = pm["rozklad_efektu_pretreningu"]
        for key, lab in (("rozdzielczosc_K36p16_minus_K4", "Model 2: SAMA rozdzielczość kątowa zadania"),
                         ("ilosc_par_K36_minus_K36p16", "Model 2: SAMA liczba par"),
                         ("laczny_K36_minus_K4", "Model 2: efekt łączny K36 − K4")):
            if key in r:
                add("wyniki_echo", lab, round(r[key], 5), "RMSE", "[Z]", "pretext/summary.json",
                    "Model 2", "wartość ujemna = poprawa")

    # ---- 6. budzet
    b = exp.get("budzet", {})
    if b:
        add("budzet", "Przebiegów w macierzy", b.get("razem", {}).get("przebiegow"), "przebiegów", "[W]",
            "experiments.json", f"{R10} §5.5", "22 warunki × 3 ziarna")
        add("budzet", "Czas całej macierzy (--fast-bilinear)", b.get("razem", {}).get("godzin"), "h", "[W]",
            "experiments.json", f"{R10} §5.5", "dolne oszacowanie — bez narzutu walidacji")
        add("budzet", "Dysk: cała macierz", b.get("razem", {}).get("GB_z_adamem"), "GB", "[W]",
            "experiments.json", f"{R11} §1.1", "po dodaniu drugiego checkpointu val@4 (+10,6 GB na glowne)")
        nm = b.get("na_model", {})
        for k in ("full", "echo2depth"):
            if k in nm:
                add("budzet", f"Dysk na przebieg: {k}", nm[k].get("per_run_total_GB"), "GB", "[W]",
                    "experiments.json", f"{R11} §1.1", "2× wagi + checkpoint z Adamem (3× parametry)")
    if disk:
        add("budzet", "Wolne miejsce na dysku", disk.get("wolne_GB"), "GB", "[Z]",
            "disk_budget.json", f"{R11} §0")
        add("budzet", "Margines po całej macierzy", round(disk.get("wolne_GB", 0)
            - disk.get("wszystko_razem_GB", 0) - 20, 1), "GB", "[Z]", "disk_budget.json", f"{R11} §0",
            "po odjęciu zapasu 20 GB")
    add("budzet", "Czas przebiegu: pełny model / echo2depth", "0,86 / 0,13", "h", "[Z-]",
        "bench/bench_main.json", f"{R10} §5.5", "z --fast-bilinear; bez walidacji")

    # ---- 7. literatura
    add("literatura", "Gao 2020, RGB2Depth (Replica)", 0.374, "RMSE", "[W]",
        "VisualEchoes, ECCV 2020", "—", "INNY silnik akustyczny — porównanie wyłącznie wewnętrzne")
    add("literatura", "Gao 2020, RGB+Echo2Depth (Replica)", 0.346, "RMSE", "[W]",
        "VisualEchoes, ECCV 2020", "—", "wkład echa u Gao: 7,5 %")
    add("literatura", "Gao 2020, Scratch (tabela 3)", 0.360, "RMSE", "[W]",
        "VisualEchoes, ECCV 2020, tab. 3", "—", "zadanie docelowe RGB2Depth po pretreningu")
    add("literatura", "Gao 2020, SimpleVisualEchoes 2 klasy (tabela 3)", 0.340, "RMSE", "[W]",
        "VisualEchoes, ECCV 2020, tab. 3", "—")
    add("literatura", "Gao 2020, VisualEchoes 4 klasy (tabela 3)", 0.332, "RMSE", "[W]",
        "VisualEchoes, ECCV 2020, tab. 3", "—", "trend monotoniczny po liczbie klas — nasza oś K")
    add("literatura", "Gao 2020, trafność zadania pretekstowego K=4", 66.0, "%", "[W]",
        "VisualEchoes, ECCV 2020, suplement §I", "—", "poziom losowy 25 %")
    add("literatura", "Parida 2021, marginalny wkład echa", "NIE RAPORTOWANY", "—", "[W]",
        "Beyond Image to Depth, CVPR 2021", f"{R10} §3.3",
        "dlatego trzeba go było zmierzyć samodzielnie")

    # ---- czego brakuje
    BRAK = [
        {"czego": "Krzywa nasycenia na NATURALNEJ liczności (C6/C9/C12/C18)",
         "da_to": "grupa `krzywa`, 12 przebiegow, 10,4 h",
         "po_co": "odsunieta: rosnie po gestosci I rozmiarze zbioru naraz -- krzywa stalego budzetu jest ostrzejsza"},
        {"czego": "Rozrzut po ziarnach dla PELNEGO modelu",
         "da_to": "A/B/D x 3 ziarna (odwolane degradacja 2026-08-11 §2)",
         "po_co": "liczby A/B/D maja n=1; podloga szumu zmierzona tylko na warunku A"},
        {"czego": "Delta(main vs patched) na masce SCISLEJ",
         "da_to": "evaluate.py --intersection-mask na EPA/EPB/EPD (juz sa checkpointy)",
         "po_co": "zamkniecie zastrzezenia o pikselach zmienionych a waznych"},
        {"czego": "Diagnoza NEGATYWNEGO transferu Modelu 2",
         "da_to": "porownanie wag enkodera przed/po pretreningu",
         "po_co": "czy enkoder w ogole sie uczy, czy zamiera na trywialnym rozwiazaniu"},
        {"czego": "c_full — całkowity wkład echa w PEŁNYM modelu", "da_to": "SE + B, ziarno 0",
         "po_co": "górne ograniczenie na efekt gęstości w warunkach A/B/D; decyduje o liczbie ziaren grupy glowne"},
        {"czego": "Krzywa nasycenia 4/6/9/12/18/36", "da_to": "grupa `krzywa` (C6/C9/C12/C18)",
         "po_co": "kształt zależności od gęstości, nie tylko dwa końce"},
        {"czego": "Transfer geometrii na office_4", "da_to": "dowolny warunek `patched` + `main`",
         "po_co": "sonda przy danych testowych trzymanych dosłownie stałych"},
        {"czego": "Rozrzut po ziarnach dla pełnego modelu", "da_to": "dowolny warunek `glowne` × 3 ziarna",
         "po_co": "podłoga szumu zmierzona na warunku A; nie wiadomo, czy przenosi się na inne"},
    ]
    return N, BRAK


def render_md(N: list[dict], BRAK: list[dict]) -> str:
    L = ["# Liczby do pracy — zestawienie zmierzonych wartości",
         "",
         "Wygenerowane automatycznie przez `my-operations/ml/thesis_numbers.py`. Nie edytować ręcznie —",
         "zmiany przepadną przy następnym uruchomieniu. Źródło maszynowe: `outputs/ml/thesis_numbers.json`.",
         "",
         "| status | znaczenie |", "|---|---|",
         "| **[Z]** | zmierzone: skrypt, surowe wyjście, liczba w dokumencie |",
         "| **[Z-]** | zmierzone z zastrzeżeniem, które trzeba cytować razem z liczbą |",
         "| **[W]** | wywnioskowane z kodu/źródła, nie z pomiaru |",
         ""]
    for key, title in GRUPY:
        rows = [n for n in N if n["grupa"] == key]
        if not rows:
            continue
        L += [f"## {title}", ""]
        if key == "literatura":
            L += ["> **Te liczby NIE są naszymi pomiarami.** Silnik akustyczny, przetwarzanie scen",
                  "> i zbiór lokalizacji są inne. Służą wyłącznie do sprawdzenia, czy odtwarzamy",
                  "> właściwy **porządek** warunków i **rząd wielkości** efektu — nigdy do bezpośredniego",
                  "> zestawienia w jednej kolumnie z naszymi wynikami.", ""]
        L += ["| wielkość | wartość | jedn. | status | dowód | sekcja |", "|---|---|---|---|---|---|"]
        for n in rows:
            v = n["wartosc"]
            if isinstance(v, float):
                v = f"{v:.5f}".rstrip("0").rstrip(".") if abs(v) >= 1e-4 else f"{v:.3e}"
            elif isinstance(v, dict):
                v = ", ".join(f"{k}: {vv}" for k, vv in v.items())
            L.append(f"| {n['nazwa']} | **{v}** | {n['jednostka']} | {n['status']} | "
                     f"`{n['dowod']}` | {n['sekcja']} |")
        L.append("")
        notes = [n for n in rows if n["uwaga"]]
        if notes:
            L.append("Uwagi:")
            L += [f"- **{n['nazwa']}** — {n['uwaga']}" for n in notes]
            L.append("")
    L += ["---", "", "## Liczby, których jeszcze NIE MA", "",
          "Zostaw w tekście lukę i wróć, gdy odpowiedni warunek się policzy.", "",
          "| czego brakuje | da to | po co |", "|---|---|---|"]
    L += [f"| {b['czego']} | {b['da_to']} | {b['po_co']} |" for b in BRAK]
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    N, BRAK = collect()
    out_json = paths.ML_OUTPUTS / "thesis_numbers.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"liczby": N, "brakujace": BRAK,
         "legenda": {"[Z]": "zmierzone", "[Z-]": "zmierzone z zastrzeżeniem",
                     "[W]": "wywnioskowane, nie zmierzone"},
         "uwaga_literatura": "grupa `literatura` to NIE nasze pomiary -- inny silnik akustyczny; "
                             "porownanie wylacznie wewnetrzne (porzadek i rzad wielkosci)"},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md = paths.MY_OPERATIONS / "docs" / "LICZBY_DO_PRACY.md"
    out_md.write_text(render_md(N, BRAK), encoding="utf-8")
    print(f"pozycji: {len(N)}  (w tym literatura: {sum(1 for n in N if n['grupa']=='literatura')})")
    print(f"brakujacych: {len(BRAK)}")
    for key, title in GRUPY:
        print(f"  {title:52s} {sum(1 for n in N if n['grupa']==key):3d}")
    print(f"\nzapisano: {out_json}\n          {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
