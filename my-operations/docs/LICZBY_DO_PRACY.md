# Liczby do pracy — zestawienie zmierzonych wartości

Wygenerowane automatycznie przez `my-operations/ml/thesis_numbers.py`. Nie edytować ręcznie —
zmiany przepadną przy następnym uruchomieniu. Źródło maszynowe: `outputs/ml/thesis_numbers.json`.

| status | znaczenie |
|---|---|
| **[Z]** | zmierzone: skrypt, surowe wyjście, liczba w dokumencie |
| **[Z-]** | zmierzone z zastrzeżeniem, które trzeba cytować razem z liczbą |
| **[W]** | wywnioskowane z kodu/źródła, nie z pomiaru |

## 1. Zbiór danych

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| Scen Replica łącznie | **18** | scen | [Z] | `splits/replica_locations.json` | RAPORT_SESJI_2026-08-05.md §3.4 |
| Scen treningowych / held-out | **15 / 3** | scen | [Z] | `splits/replica_locations.json` | RAPORT_SESJI_2026-08-05.md §3.2 |
| Lokalizacji train / val / test | **1374 / 183 / 183** | lokalizacji | [Z] | `splits/replica_locations.json` | RAPORT_SESJI_2026-08-05.md §3.2 |
| Odcisk podziału | **e0bf7547668d9e0a** | sha256[:16] | [Z] | `splits/replica_locations.json` | RAPORT_SESJI_2026-08-05.md §3.2 |
| Orientacji na lokalizację | **36** | kątów co 10° | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §1 |
| Próbek łącznie, wariant main | **62640** | próbek | [Z] | `verify_loader/main/verify_loader.json` | RAPORT_SESJI_2026-08-05.md §3.4 |
| Próbek łącznie, wariant patched | **44064** | próbek | [Z] | `docs/GENERATOR_PARAMS.md §4.5` | docs/GENERATOR_PARAMS.md §4.5 |
| Kształt spektrogramu | **(2, 257, 166)** | kanały × f × t | [Z] | `atrybuty HDF5` | RAPORT_SESJI_2026-08-05.md §3.1 |
| max_depth (Replica) | **14.104** | m | [Z] | `base_options.py Paridy` | RAPORT_SESJI_2026-08-05.md §3.5 |
| Pikseli powyżej max_depth | **131 (1,3·10⁻⁵ %)** | pikseli | [Z] | `verify_loader/main/verify_loader.json` | RAPORT_SESJI_2026-08-05.md §3.5 |
| Udział pikseli depth == 0, main | **8.48034** | % | [Z] | `verify_loader/main/verify_loader.json` | RAPORT_SESJI_2026-08-10.md §1.1 |
| Udział pikseli depth == 0, patched | **0.21207** | % | [Z] | `verify_loader/patched/verify_loader.json` | RAPORT_SESJI_2026-08-10.md §1.1 |
| Udział pikseli krawędziowych (próg 0,10 m/px) | **11.3** | % | [Z] | `eval/*/eval.json` | RAPORT_SESJI_2026-08-05.md §3.12 |

Uwagi:
- **Scen treningowych / held-out** — held-out: apartment_2, frl_apartment_5, office_4
- **Odcisk podziału** — ZAMROŻONY — nie regenerować
- **Próbek łącznie, wariant patched** — 10 scen łatanych; do treningu + 8 scen szczelnych z main
- **Kształt spektrogramu** — bit-zgodny z generate_spectrogram() Paridy
- **Pikseli powyżej max_depth** — wszystkie w apartment_0 (scena treningowa)
- **Udział pikseli krawędziowych (próg 0,10 m/px)** — 0,05 m/px → 21,54 %; 0,20 m/px → 5,86 %

## 2. Charakterystyka silnika akustycznego

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| indirectRayCount | **500** | promieni | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §1 |
| threadCount | **1** | wątek | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §1 |
| Sygnał przy 10° (SIGNAL_10DEG) | **0.0644** | RMSE spektrogramu | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §3 |
| Docelowy SNR | **3.5** | — | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §3 |
| Szum render-do-renderu (ten sam kąt) | **0,03–0,16** | RMSE spektrogramu | [Z] | `diagnose_rlr_noise_out/` | CLAUDE.md |
| RMSE między renderami 90° od siebie | **0,30–0,35** | RMSE spektrogramu | [Z] | `diagnose_rlr_noise_out/` | CLAUDE.md |
| N adaptacyjne: min / max / próbne | **6 / 64 / 8** | renderów | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §3 |
| Czas renderu (materiały włączone) | **0,111–0,148** | s/render | [Z] | `atrybuty HDF5` | docs/GENERATOR_PARAMS.md §4.3 |
| Przepustowość dataloadera (8 workerów) | **2645.1** | próbek/s | [Z] | `bench/bench_main.json` | RAPORT_SESJI_2026-08-05.md §3.7 |

Uwagi:
- **threadCount** — wątki dzielą budżet promieni
- **Szum render-do-renderu (ten sam kąt)** — zależny od pozycji
- **RMSE między renderami 90° od siebie** — potwierdzone niezależnie w geometry_check: 0,3029–0,3159

## 3. Geometria `main` vs `patched`

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| Zmienionych wartości depth (frl_apartment_5) | **17.571** | % wartości | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.1 |
| Zmienionych wartości echo (wszystkie sceny) | **91,9–93,2** | % komórek | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.1 |
| Pikseli usuniętych przez łatkę (+ → 0) | **0** | pikseli | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.2 |
| Energia całkowita: otwarte vs szczelne | **-8.24327** | % | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.5 |
| Energia PÓŹNA (pogłos): otwarte vs szczelne | **-52.21371** | % | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.5 |
| Względny kontrast kątowy późny: otwarte / szczelne | **1.16854** | × | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.6 |
| Energia PÓŹNA: patched vs main | **46.33968** | % | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.5 |
| Względny kontrast kątowy późny: patched / main | **0.82508** | × | [Z] | `geometry_check/geometry_check.json` | RAPORT_SESJI_2026-08-10.md §2.6 |
| Granica części późnej spektrogramu | **ramka 30 (10,9 ms)** | — | [W] | `geometry_check.py::LATE_FRAME_START` | RAPORT_SESJI_2026-08-10.md §2.5 |
| Piksele zmienione, a ważne w obu wariantach | **1.599** | % kadru | [Z] | `mask_check/mask_check.json` | RAPORT_SESJI_2026-08-11.md §5 |
| Narzut maski przecięcia w ewaluacji | **37** | % | [Z] | `mask_check/mask_check.json` | RAPORT_SESJI_2026-08-11.md §5 |
| Wpływ domknięcia geometrii (echo2depth, patched − main) | **A: +0.01462, B: +0.01014, D: +0.01507** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §3 |

Uwagi:
- **Pikseli usuniętych przez łatkę (+ → 0)** — we WSZYSTKICH 10 scenach — dlatego maska przecięcia = maska main
- **Energia PÓŹNA (pogłos): otwarte vs szczelne** — to jest właściwa liczba — całkowita jest zdominowana przez ścieżkę bezpośrednią
- **Względny kontrast kątowy późny: patched / main** — domknięcie sufitu OBNIŻA kontrast kątowy przy jednoczesnym wzroście SNR — kompromis
- **Granica części późnej spektrogramu** — za pierwszym odbiciem podłoga/sufit przy 1,25 m (7,3 ms)
- **Wpływ domknięcia geometrii (echo2depth, patched − main)** — 1 ziarno; wartości DODATNIE = `patched` GORSZY mimo +46 % energii pogłosu

## 4. Determinizm i wydajność

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| Podłoga szumu frameworka (|ΔRMSE| po 2000 krokach) | **0.00732** | RMSE | [Z] | `determinism/determinism_check.json` | RAPORT_SESJI_2026-08-10.md §3.1 |
| Rozbieżność podstawienia BilinearEinsum (|ΔRMSE|) | **0.00073** | RMSE | [Z] | `determinism/determinism_check.json` | RAPORT_SESJI_2026-08-10.md §3.1 |
| Rozbieżność wag: podłoga / podstawienie | **1.022e-02 / 1.168e-02** | względna L2 | [Z] | `determinism/determinism_check.json` | RAPORT_SESJI_2026-08-10.md §3.1 |
| Wagi startowe nn.Bilinear vs BilinearEinsum | **bit-identyczne** | — | [Z] | `determinism/determinism_check.json` | RAPORT_SESJI_2026-08-10.md §3.1 |
| Przyspieszenie --fast-bilinear (pełna pętla) | **16.15** | × | [Z] | `determinism/determinism_check.json` | RAPORT_SESJI_2026-08-10.md §3.1 |
| Parametry: pełny model | **316918781** | parametrów | [Z] | `experiments.py::PARAM_COUNTS` | RAPORT_SESJI_2026-08-10.md §5.5 |
| Parametry: echo2depth | **8984073** | parametrów | [Z] | `experiments.py::PARAM_COUNTS` | RAPORT_SESJI_2026-08-10.md §5.5 |
| Parametry: Model 2 (pretekst) | **25733446** | parametrów | [Z] | `pretext/model.py` | RAPORT_SESJI_2026-08-10.md §6.2 |
| Zgodność metryk z implementacją Paridy | **1.494e-06** | max |różnica| | [Z] | `metrics.py::test_matches_parida()` | RAPORT_SESJI_2026-08-05.md §3.11 |
| Zgodność tabeli per próbka z akumulatorem | **2.745e-08** | max |różnica| | [Z] | `metrics.py::test_table_matches_accumulator()` | RAPORT_SESJI_2026-08-10.md §4 |

Uwagi:
- **Podłoga szumu frameworka (|ΔRMSE| po 2000 krokach)** — dwa przebiegi TEGO SAMEGO kodu, to samo ziarno; zakres po krokach 0,0021–0,0096
- **Rozbieżność podstawienia BilinearEinsum (|ΔRMSE|)** — 10× PONIŻEJ podłogi
- **Przyspieszenie --fast-bilinear (pełna pętla)** — 1,5391 → 0,0953 s/krok; mikrobenchmark samego kroku dawał 19,5×
- **Parametry: pełny model** — rgbdepth 16 658 561 + audio 8 984 073 + attention 279 581 505 + material 11 694 642
- **Parametry: Model 2 (pretekst)** — w tym RGBDepthNet 16 658 561 do przeniesienia
- **Zgodność tabeli per próbka z akumulatorem** — 17 kontroli

## 5. Wyniki grupy `echo`

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| RMSE test@36: EA | **0.79104 ± 0.01066** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| RMSE test@36: ED | **0.64432 ± 0.00238** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| RMSE test@36: EB | **0.58223 ± 0.00348** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| RMSE test@36: ESE | **1.16292 ± 0.00362** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Kontrast: gestosc katowa D-A | **0.14672 ± 0.01303** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Kontrast: ilosc danych B-D | **0.06209 ± 0.00435** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Kontrast: laczny B-A | **0.20882 ± 0.01132** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Kontrast: wklad echa | **0.58070 ± 0.00076** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Udział gęstości kątowej w efekcie łącznym | **70.2 ± 3.0** | % | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Luka generalizacji kątowej EA (0° → 40°) | **0.31477 ± 0.02860** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Krzywa RMSE(odległość kątowa), EA | **0: 0.59626, 10: 0.66296, 20: 0.78249, 30: 0.87034, 40: 0.91103** | RMSE per kubełek | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Luka test@36 − test@4 per warunek | **EA: 0.19478, ED: 0.00691, EB: 0.00384, ESE: 0.00395** | RMSE | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| Rozrzut po ziarnach: EA vs pozostałe | **0,01066 wobec 0,0024–0,0036** | RMSE (sd) | [Z] | `echo_ablation/echo_3seeds.json` | RAPORT_SESJI_2026-08-11.md §3 |
| c_full — wkład echa w PEŁNYM modelu | **0.02228** | RMSE | [Z-] | `echo_ablation/full_model_gate.json` | RAPORT_SESJI_2026-08-11.md §2 |
| Względny wkład echa w pełnym modelu | **9.2** | % | [Z-] | `echo_ablation/full_model_gate.json` | RAPORT_SESJI_2026-08-11.md §2 |
| EA vs EB na test@4 (nowy protokół, 3 ziarna) | **0,01787 ± 0,01128** | RMSE | [Z] | `eval/compare_EA_seed*_vs_EB_seed*_test4.json` | RAPORT_SESJI_2026-08-11.md §4.1 |
| Całkowity wkład echa (echo2depth, walidacja) | **0.6074** | RMSE | [Z-] | `echo_ablation/echo_ablation.json` | RAPORT_SESJI_2026-08-10.md §3.3 |
| Luka test@36 − test@4 per warunek | **EA: 0.22473, ED: 0.00093, EB: 0.00141, ESE: 0.00399** | RMSE | [Z-] | `echo_ablation/gap_table_seed0.json` | RAPORT_SESJI_2026-08-11.md §4.2 |
| EA vs EB na test@4 (sparowane, te same 732 próbki) | **0.0015** | RMSE | [Z-] | `eval/compare_EA_seed0_vs_EB_seed0_test4.json` | RAPORT_SESJI_2026-08-11.md §4.1 |
| KRZYWA STAŁEGO BUDŻETU: RMSE w funkcji siatki K | **K=4: 0.79104, K=6: 0.70623, K=9: 0.66342, K=12: 0.65331, K=18: 0.64804, K=36: 0.64432** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| Nasycenie krzywej stałego budżetu | **4→9: 0.128 · 9→36: 0.019** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| Krzywa stałego budżetu: K=4 → K=36 | **0.132** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| PEŁNY MODEL: RMSE test@36 (A / B / D) | **A: 0.28739, B: 0.24205, D: 0.26909** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §2 |
| Pełny model: gęstość (D−A) / ilość danych (B−D) | **-0.01831 / -0.02703** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §2 |
| Model 2: MAAE zadania pretekstowego | **K4: 61.23, K12: 55.73, K36: 25.13, K36_p16: 61.77** | stopnie | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: rozkład efektu pretreningu | **ilość par -36.64° · rozdzielczość +0.54°** | stopnie | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: transfer RGB2Depth (5 ziaren) | **pretext_K4_seed0: 0.28699, pretext_K36_p16_seed0: 0.28927, scratch: 0.28986, pretext_K36_seed0: 0.29439, pretext_K12_seed0: 0.29688** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K12_seed0 vs scratch | **+0.00702 (p=0.074)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K36_p16_seed0 vs scratch | **-0.00059 (p=0.751)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K36_seed0 vs scratch | **+0.00453 (p=0.207)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K4_seed0 vs scratch | **-0.00287 (p=0.231)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: MAAE zadania pretekstowego, K=4 | **61.23** | stopnie | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: MAAE zadania pretekstowego, K=12 | **55.73** | stopnie | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: MAAE zadania pretekstowego, K=36 | **43.45 ± 25.91** | stopnie | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: RGB2Depth po pretreningu — pretext_K4_seed0 | **0.28699 ± 0.00433** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_p16_seed0 | **0.28927 ± 0.00340** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: RGB2Depth po pretreningu — scratch | **0.28986 ± 0.00204** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_seed0 | **0.29439 ± 0.00664** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: RGB2Depth po pretreningu — pretext_K12_seed0 | **0.29688 ± 0.00657** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: SAMA rozdzielczość kątowa zadania | **0.00228** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: SAMA liczba par | **0.00512** | RMSE | [Z] | `pretext/summary.json` | Model 2 |
| Model 2: efekt łączny K36 − K4 | **0.0074** | RMSE | [Z] | `pretext/summary.json` | Model 2 |

Uwagi:
- **RMSE test@36: EA** — średnia ± sd po 3 ziarnach
- **RMSE test@36: ED** — średnia ± sd po 3 ziarnach
- **RMSE test@36: EB** — średnia ± sd po 3 ziarnach
- **RMSE test@36: ESE** — średnia ± sd po 3 ziarnach
- **Kontrast: gestosc katowa D-A** — sd po ZIARNACH; istotne we wszystkich 3: True; CI po lokalizacjach (ziarno 0) [0.11231236253503095, 0.1517812154830765]
- **Kontrast: ilosc danych B-D** — sd po ZIARNACH; istotne we wszystkich 3: True; CI po lokalizacjach (ziarno 0) [0.0512386921897015, 0.07880838719573234]
- **Kontrast: laczny B-A** — sd po ZIARNACH; istotne we wszystkich 3: True; CI po lokalizacjach (ziarno 0) [0.1697141469308697, 0.2254509038481209]
- **Kontrast: wklad echa** — sd po ZIARNACH; istotne we wszystkich 3: True; CI po lokalizacjach (ziarno 0) [0.5404420619067919, 0.6240416489197484]
- **Udział gęstości kątowej w efekcie łącznym** — gęstość 2,36× większa niż ilość danych
- **Luka generalizacji kątowej EA (0° → 40°)** — 52.84 %, monotoniczna w 3/3 ziarnach; stary protokół dawał +0,35540 — zawyżał o 0.04063
- **Krzywa RMSE(odległość kątowa), EA** — kubełki 0/10/20/30/40°; 45° nie występuje przy siatce co 10°
- **Luka test@36 − test@4 per warunek** — lukę ma WYŁĄCZNIE warunek bez pokrycia kątowego
- **Rozrzut po ziarnach: EA vs pozostałe** — warunek o najrzadszym pokryciu kątowym jest 3–4× wrażliwszy na inicjalizację
- **c_full — wkład echa w PEŁNYM modelu** — 95 % CI [0.0184, 0.02643], ziarno 0; 26,4× mniej niż w echo2depth
- **Względny wkład echa w pełnym modelu** — u Gao 7,5 % — zgodność rzędu wielkości potwierdza poprawność potoku, NIE jest zestawieniem wyników
- **EA vs EB na test@4 (nowy protokół, 3 ziarna)** — 91,4 % kary EA powstaje na kątach NIEWIDZIANYCH
- **Całkowity wkład echa (echo2depth, walidacja)** — ziarno 0
- **Luka test@36 − test@4 per warunek** — lukę ma WYŁĄCZNIE warunek bez pokrycia kątowego
- **EA vs EB na test@4 (sparowane, te same 732 próbki)** — 95 % CI [−0,01325; +0,01731] — OBEJMUJE ZERO
- **KRZYWA STAŁEGO BUDŻETU: RMSE w funkcji siatki K** — echo2depth, 4 próbki/lokalizację (5 496) w KAŻDYM punkcie — liczność stała, zmienia się wyłącznie siatka; 3 ziarna
- **Nasycenie krzywej stałego budżetu** — przejście 4→9 daje 6,7× więcej niż 9→36 — punkt odcięcia ok. K = 9–12
- **Krzywa stałego budżetu: K=4 → K=36** — 95 % CI [0.11231; 0.15178], bootstrap po lokalizacjach
- **PEŁNY MODEL: RMSE test@36 (A / B / D)** — 1 ziarno (degradacja 2026-08-11 §2) — bez oszacowania rozrzutu po ziarnach
- **Pełny model: gęstość (D−A) / ilość danych (B−D)** — n=1 ziarno; oba porównywalne z podłogą szumu 0,0023–0,0073 — patrz zastrzeżenie
- **Model 2: MAAE zadania pretekstowego** — poziom losowy 90° NIEZALEŻNIE od K
- **Model 2: rozkład efektu pretreningu** — CAŁA przewaga K=36 pochodzi z 81× większej liczby par, NIE z rozdzielczości kątowej
- **Model 2: transfer RGB2Depth (5 ziaren)** — WYNIK NEGATYWNY — żadna różnica wobec `scratch` nie jest istotna
- **Model 2: transfer pretext_K12_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K36_p16_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K36_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K4_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: MAAE zadania pretekstowego, K=4** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1
- **Model 2: MAAE zadania pretekstowego, K=12** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1
- **Model 2: MAAE zadania pretekstowego, K=36** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1
- **Model 2: RGB2Depth po pretreningu — pretext_K4_seed0** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu
- **Model 2: RGB2Depth po pretreningu — pretext_K36_p16_seed0** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu
- **Model 2: RGB2Depth po pretreningu — scratch** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu
- **Model 2: RGB2Depth po pretreningu — pretext_K36_seed0** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu
- **Model 2: RGB2Depth po pretreningu — pretext_K12_seed0** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu
- **Model 2: SAMA rozdzielczość kątowa zadania** — wartość ujemna = poprawa
- **Model 2: SAMA liczba par** — wartość ujemna = poprawa
- **Model 2: efekt łączny K36 − K4** — wartość ujemna = poprawa

## 6. Budżet obliczeniowy i dyskowy

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| Przebiegów w macierzy | **66** | przebiegów | [W] | `experiments.json` | RAPORT_SESJI_2026-08-10.md §5.5 |
| Czas całej macierzy (--fast-bilinear) | **32.8** | h | [W] | `experiments.json` | RAPORT_SESJI_2026-08-10.md §5.5 |
| Dysk: cała macierz | **200.2** | GB | [W] | `experiments.json` | RAPORT_SESJI_2026-08-11.md §1.1 |
| Dysk na przebieg: full | **5.903** | GB | [W] | `experiments.json` | RAPORT_SESJI_2026-08-11.md §1.1 |
| Dysk na przebieg: echo2depth | **0.167** | GB | [W] | `experiments.json` | RAPORT_SESJI_2026-08-11.md §1.1 |
| Wolne miejsce na dysku | **227.1** | GB | [Z] | `disk_budget.json` | RAPORT_SESJI_2026-08-11.md §0 |
| Margines po całej macierzy | **5.4** | GB | [Z] | `disk_budget.json` | RAPORT_SESJI_2026-08-11.md §0 |
| Czas przebiegu: pełny model / echo2depth | **0,86 / 0,13** | h | [Z-] | `bench/bench_main.json` | RAPORT_SESJI_2026-08-10.md §5.5 |

Uwagi:
- **Przebiegów w macierzy** — 22 warunki × 3 ziarna
- **Czas całej macierzy (--fast-bilinear)** — dolne oszacowanie — bez narzutu walidacji
- **Dysk: cała macierz** — po dodaniu drugiego checkpointu val@4 (+10,6 GB na glowne)
- **Dysk na przebieg: full** — 2× wagi + checkpoint z Adamem (3× parametry)
- **Dysk na przebieg: echo2depth** — 2× wagi + checkpoint z Adamem (3× parametry)
- **Margines po całej macierzy** — po odjęciu zapasu 20 GB
- **Czas przebiegu: pełny model / echo2depth** — z --fast-bilinear; bez walidacji

## 7. Odniesienia z literatury — NIE nasze pomiary

> **Te liczby NIE są naszymi pomiarami.** Silnik akustyczny, przetwarzanie scen
> i zbiór lokalizacji są inne. Służą wyłącznie do sprawdzenia, czy odtwarzamy
> właściwy **porządek** warunków i **rząd wielkości** efektu — nigdy do bezpośredniego
> zestawienia w jednej kolumnie z naszymi wynikami.

| wielkość | wartość | jedn. | status | dowód | sekcja |
|---|---|---|---|---|---|
| Gao 2020, RGB2Depth (Replica) | **0.374** | RMSE | [W] | `VisualEchoes, ECCV 2020` | — |
| Gao 2020, RGB+Echo2Depth (Replica) | **0.346** | RMSE | [W] | `VisualEchoes, ECCV 2020` | — |
| Gao 2020, Scratch (tabela 3) | **0.36** | RMSE | [W] | `VisualEchoes, ECCV 2020, tab. 3` | — |
| Gao 2020, SimpleVisualEchoes 2 klasy (tabela 3) | **0.34** | RMSE | [W] | `VisualEchoes, ECCV 2020, tab. 3` | — |
| Gao 2020, VisualEchoes 4 klasy (tabela 3) | **0.332** | RMSE | [W] | `VisualEchoes, ECCV 2020, tab. 3` | — |
| Gao 2020, trafność zadania pretekstowego K=4 | **66** | % | [W] | `VisualEchoes, ECCV 2020, suplement §I` | — |
| Parida 2021, marginalny wkład echa | **NIE RAPORTOWANY** | — | [W] | `Beyond Image to Depth, CVPR 2021` | RAPORT_SESJI_2026-08-10.md §3.3 |

Uwagi:
- **Gao 2020, RGB2Depth (Replica)** — INNY silnik akustyczny — porównanie wyłącznie wewnętrzne
- **Gao 2020, RGB+Echo2Depth (Replica)** — wkład echa u Gao: 7,5 %
- **Gao 2020, Scratch (tabela 3)** — zadanie docelowe RGB2Depth po pretreningu
- **Gao 2020, VisualEchoes 4 klasy (tabela 3)** — trend monotoniczny po liczbie klas — nasza oś K
- **Gao 2020, trafność zadania pretekstowego K=4** — poziom losowy 25 %
- **Parida 2021, marginalny wkład echa** — dlatego trzeba go było zmierzyć samodzielnie

---

## Liczby, których jeszcze NIE MA

Zostaw w tekście lukę i wróć, gdy odpowiedni warunek się policzy.

| czego brakuje | da to | po co |
|---|---|---|
| Krzywa nasycenia na NATURALNEJ liczności (C6/C9/C12/C18) | grupa `krzywa`, 12 przebiegow, 10,4 h | odsunieta: rosnie po gestosci I rozmiarze zbioru naraz -- krzywa stalego budzetu jest ostrzejsza |
| Rozrzut po ziarnach dla PELNEGO modelu | A/B/D x 3 ziarna (odwolane degradacja 2026-08-11 §2) | liczby A/B/D maja n=1; podloga szumu zmierzona tylko na warunku A |
| Delta(main vs patched) na masce SCISLEJ | evaluate.py --intersection-mask na EPA/EPB/EPD (juz sa checkpointy) | zamkniecie zastrzezenia o pikselach zmienionych a waznych |
| Diagnoza NEGATYWNEGO transferu Modelu 2 | porownanie wag enkodera przed/po pretreningu | czy enkoder w ogole sie uczy, czy zamiera na trywialnym rozwiazaniu |
| c_full — całkowity wkład echa w PEŁNYM modelu | SE + B, ziarno 0 | górne ograniczenie na efekt gęstości w warunkach A/B/D; decyduje o liczbie ziaren grupy glowne |
| Krzywa nasycenia 4/6/9/12/18/36 | grupa `krzywa` (C6/C9/C12/C18) | kształt zależności od gęstości, nie tylko dwa końce |
| Transfer geometrii na office_4 | dowolny warunek `patched` + `main` | sonda przy danych testowych trzymanych dosłownie stałych |
| Rozrzut po ziarnach dla pełnego modelu | dowolny warunek `glowne` × 3 ziarna | podłoga szumu zmierzona na warunku A; nie wiadomo, czy przenosi się na inne |
