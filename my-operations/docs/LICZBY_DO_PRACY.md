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
| Wpływ domknięcia geometrii (patched − main) — 1 ziarno, ZASTĄPIONE | **A: +0.01462, B: +0.01014, D: +0.01507** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §3 |
| Efekt gęstości `gestosc_D_minus_A`: main vs patched | **main -0.14672 · patched -0.13504 · różnica +0.01168** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Efekt gęstości `laczny_B_minus_A`: main vs patched | **main -0.20882 · patched -0.19524 · różnica +0.01358** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Efekt gęstości `ilosc_danych_B_minus_D`: main vs patched | **main -0.06209 · patched -0.06019 · różnica +0.00190** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Wpływ domknięcia geometrii: EPA_minus_EA (3 ziarna) | **-0.00123 (p=0.870)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Wpływ domknięcia geometrii: EPB_minus_EB (3 ziarna) | **+0.01235 (p=0.013)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Wpływ domknięcia geometrii: EPD_minus_ED (3 ziarna) | **+0.01045 (p=0.090)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| Δ na trzech maskach: EPA_minus_EA (3 ziarna) | **pelna: -0.00123 ± 0.01389, intersection: -0.00429 ± 0.01088, strict: +0.00389 ± 0.00981** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1b |
| Δ na trzech maskach: EPB_minus_EB (3 ziarna) | **pelna: +0.01235 ± 0.00285, intersection: +0.00325 ± 0.00309, strict: +0.00704 ± 0.00326** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1b |
| Δ na trzech maskach: EPD_minus_ED (3 ziarna) | **pelna: +0.01045 ± 0.00402, intersection: +0.00299 ± 0.00496, strict: +0.00826 ± 0.00520** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1b |
| Czy Δ(patched−main) jest dodatnia we wszystkich 9 komórkach | **False** | — | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1b |

Uwagi:
- **Pikseli usuniętych przez łatkę (+ → 0)** — we WSZYSTKICH 10 scenach — dlatego maska przecięcia = maska main
- **Energia PÓŹNA (pogłos): otwarte vs szczelne** — to jest właściwa liczba — całkowita jest zdominowana przez ścieżkę bezpośrednią
- **Względny kontrast kątowy późny: patched / main** — domknięcie sufitu OBNIŻA kontrast kątowy przy jednoczesnym wzroście SNR — kompromis
- **Granica części późnej spektrogramu** — za pierwszym odbiciem podłoga/sufit przy 1,25 m (7,3 ms)
- **Wpływ domknięcia geometrii (patched − main) — 1 ziarno, ZASTĄPIONE** — ZASTĄPIONE: na 3 ziarnach `EPA − EA` ZMIENIA ZNAK (−0,00123, p = 0,87), więc teza o pogorszeniu we wszystkich trzech warunkach NIE utrzymuje się; RAPORT_SESJI_2026-08-15.md §3.1
- **Efekt gęstości `gestosc_D_minus_A`: main vs patched** — p=0.259 (Welch po ziarnach); znak zgodny w obu geometriach: True. To jest WŁAŚCIWA wielkość porównywana między wariantami — surowe RMSE liczą się na różnych zbiorach pikseli ważnych
- **Efekt gęstości `laczny_B_minus_A`: main vs patched** — p=0.162 (Welch po ziarnach); znak zgodny w obu geometriach: True. To jest WŁAŚCIWA wielkość porównywana między wariantami — surowe RMSE liczą się na różnych zbiorach pikseli ważnych
- **Efekt gęstości `ilosc_danych_B_minus_D`: main vs patched** — p=0.736 (Welch po ziarnach); znak zgodny w obu geometriach: True. To jest WŁAŚCIWA wielkość porównywana między wariantami — surowe RMSE liczą się na różnych zbiorach pikseli ważnych
- **Wpływ domknięcia geometrii: EPA_minus_EA (3 ziarna)** — wartość DODATNIA = `patched` GORSZY; 0.2-0.5x podlogi szumu
- **Wpływ domknięcia geometrii: EPB_minus_EB (3 ziarna)** — wartość DODATNIA = `patched` GORSZY; 1.7-5.4x podlogi szumu
- **Wpływ domknięcia geometrii: EPD_minus_ED (3 ziarna)** — wartość DODATNIA = `patched` GORSZY; 1.4-4.5x podlogi szumu
- **Δ na trzech maskach: EPA_minus_EA (3 ziarna)** — maska `pelna` punktuje każdy wariant na JEGO pikselach ważnych i dlatego zawyża Δ; `intersection` i `strict` liczą oba na tych samych pikselach
- **Δ na trzech maskach: EPB_minus_EB (3 ziarna)** — maska `pelna` punktuje każdy wariant na JEGO pikselach ważnych i dlatego zawyża Δ; `intersection` i `strict` liczą oba na tych samych pikselach
- **Δ na trzech maskach: EPD_minus_ED (3 ziarna)** — maska `pelna` punktuje każdy wariant na JEGO pikselach ważnych i dlatego zawyża Δ; `intersection` i `strict` liczą oba na tych samych pikselach
- **Czy Δ(patched−main) jest dodatnia we wszystkich 9 komórkach** — na 1 ziarnie (2026-08-13 §3.1) było 9/9 — na 3 ziarnach NIE; `EPA` jest nierozróżnialne od zera na każdej masce

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
| Luka test@36 − test@4 per warunek (1 ziarno, stary protokół) | **EA: 0.22473, ED: 0.00093, EB: 0.00141, ESE: 0.00399** | RMSE | [Z-] | `echo_ablation/gap_table_seed0.json` | RAPORT_SESJI_2026-08-11.md §4.2 |
| EA vs EB na test@4 (sparowane, te same 732 próbki) | **0.0015** | RMSE | [Z-] | `eval/compare_EA_seed0_vs_EB_seed0_test4.json` | RAPORT_SESJI_2026-08-11.md §4.1 |
| KRZYWA STAŁEGO BUDŻETU: RMSE w funkcji siatki K | **K=4: 0.79104, K=6: 0.70623, K=9: 0.66342, K=12: 0.65331, K=18: 0.64804, K=36: 0.64432** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| Nasycenie krzywej stałego budżetu | **4→9: 0.128 · 9→36: 0.019** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| Krzywa stałego budżetu: K=4 → K=36 | **0.132** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §1 |
| PEŁNY MODEL: RMSE test@36 (A / B / D) — 1 ziarno, ZASTĄPIONE | **A: 0.28739, B: 0.24205, D: 0.26909** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §2 |
| Pełny model: gęstość / ilość danych — 1 ziarno, ZASTĄPIONE | **-0.01831 / -0.02703** | RMSE | [Z-] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §2 |
| Model 2: MAAE zadania pretekstowego | **K4: 61.23, K12: 55.73, K36: 25.13, K36_p16: 61.77** | stopnie | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: rozkład MAAE pretekstu (stopnie) | **ilość par -36.64° · rozdzielczość +0.54°** | stopnie | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: transfer RGB2Depth (5 ziaren) | **pretext_K4_seed0: 0.28699, pretext_K36_p16_seed0: 0.28927, scratch: 0.28986, pretext_K36_seed0: 0.29439, pretext_K12_seed0: 0.29688** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K12_seed0 vs scratch | **+0.00702 (p=0.074)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K36_p16_seed0 vs scratch | **-0.00059 (p=0.751)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K36_seed0 vs scratch | **+0.00453 (p=0.207)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer pretext_K4_seed0 vs scratch | **-0.00287 (p=0.231)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-13.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: transfer przy 10% zbioru treningowego | **scratch: 0.35396, pretext_K4_seed0: 0.35835, pretext_K36_seed0: 0.35390** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: transfer przy 25% zbioru treningowego | **scratch: 0.30083, pretext_K4_seed0: 0.30569, pretext_K36_seed0: 0.30516** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K4_seed0_vs_scratch@10% | **+0.00439 (p=0.540)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K36_seed0_vs_scratch@10% | **-0.00005 (p=0.990)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K4_seed0_vs_scratch@25% | **+0.00486 (p=0.255)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K36_seed0_vs_scratch@25% | **+0.00433 (p=0.156)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K4_seed0_vs_scratch@100% | **-0.00287 (p=0.231)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: pretext_K36_seed0_vs_scratch@100% | **+0.00453 (p=0.207)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2 |
| Model 2: WERDYKT przewidywania §5.1 — pretext_K4_seed0 | **OBALONE** | — | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2.4 |
| Model 2: WERDYKT przewidywania §5.1 — pretext_K36_seed0 | **OBALONE** | — | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §2.4 |
| echo2depth RMSE test@36, geometria `main` (3 ziarna) | **EA: 0.79104 ± 0.01066, EB: 0.58223 ± 0.00348, ED: 0.64432 ± 0.00238** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| echo2depth RMSE test@36, geometria `patched` (3 ziarna) | **EPA: 0.78982 ± 0.00526, EPB: 0.59458 ± 0.00173, EPD: 0.65477 ± 0.00637** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.1 |
| PEŁNY MODEL: RMSE test@36 A/B/D (3 ziarna) | **A: 0.29248 ± 0.00488, B: 0.24367 ± 0.00142, D: 0.27199 ± 0.00265** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.2 |
| Pełny model: gestosc_D_minus_A (3 ziarna) | **-0.02048 ± 0.00350 (p=0.0096)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.2 |
| Pełny model: ilosc_danych_B_minus_D (3 ziarna) | **-0.02833 ± 0.00154 (p=0.0010)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.2 |
| Pełny model: laczny_B_minus_A (3 ziarna) | **-0.04881 ± 0.00355 (p=0.0018)** | RMSE | [Z] | `echo_ablation/final_results_2026-08-15.json` | RAPORT_SESJI_2026-08-15.md §3.2 |
| Model 2: MAAE pretekstu wg wariantu — K4 | **59.94 ± 2.10** | stopnie | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: MAAE pretekstu wg wariantu — K12 | **58.73 ± 2.61** | stopnie | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: MAAE pretekstu wg wariantu — K36 | **25.65 ± 0.74** | stopnie | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: MAAE pretekstu wg wariantu — K36@16par | **58.70 ± 7.40** | stopnie | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §4 |
| Model 2: RGB2Depth po pretreningu — pretext_K4_seed0 @ 100% zbioru | **0.28699 ± 0.00433** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_p16_seed0 @ 100% zbioru | **0.28927 ± 0.00340** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — scratch @ 100% zbioru | **0.28986 ± 0.00204** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_seed0 @ 100% zbioru | **0.29439 ± 0.00664** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K12_seed0 @ 100% zbioru | **0.29688 ± 0.00657** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — scratch_f25 @ 25% zbioru | **0.30083 ± 0.00350** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_seed0_f25 @ 25% zbioru | **0.30516 ± 0.00113** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K4_seed0_f25 @ 25% zbioru | **0.30569 ± 0.00512** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K36_seed0_f10 @ 10% zbioru | **0.35390 ± 0.00559** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — scratch_f10 @ 10% zbioru | **0.35396 ± 0.00472** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RGB2Depth po pretreningu — pretext_K4_seed0_f10 @ 10% zbioru | **0.35835 ± 0.00990** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RMSE docelowego — SAMA rozdzielczość kątowa | **0.00228** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RMSE docelowego — SAMA liczba par | **0.00512** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |
| Model 2: RMSE docelowego — efekt łączny K36 − K4 | **0.0074** | RMSE | [Z] | `pretext/summary.json` | RAPORT_SESJI_2026-08-13.md §5 |

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
- **Luka test@36 − test@4 per warunek (1 ziarno, stary protokół)** — ZASTĄPIONE przez wersję z 3 ziaren (`echo_3seeds.json`) — nie cytować obu naraz
- **EA vs EB na test@4 (sparowane, te same 732 próbki)** — 95 % CI [−0,01325; +0,01731] — OBEJMUJE ZERO
- **KRZYWA STAŁEGO BUDŻETU: RMSE w funkcji siatki K** — echo2depth, 4 próbki/lokalizację (5 496) w KAŻDYM punkcie — liczność stała, zmienia się wyłącznie siatka; 3 ziarna
- **Nasycenie krzywej stałego budżetu** — przejście 4→9 daje 6,7× więcej niż 9→36 — punkt odcięcia ok. K = 9–12
- **Krzywa stałego budżetu: K=4 → K=36** — 95 % CI [0.11231; 0.15178], bootstrap po lokalizacjach
- **PEŁNY MODEL: RMSE test@36 (A / B / D) — 1 ziarno, ZASTĄPIONE** — ZASTĄPIONE przez wersję z 3 ziaren (RAPORT_SESJI_2026-08-15.md §3.2) — nie cytować
- **Pełny model: gęstość / ilość danych — 1 ziarno, ZASTĄPIONE** — ZASTĄPIONE: na 3 ziarnach gęstość wynosi −0,02048 (p = 0,0096), a nie −0,01831; patrz RAPORT_SESJI_2026-08-15.md §3.2 — nie cytować tej wersji
- **Model 2: MAAE zadania pretekstowego** — poziom losowy 90° NIEZALEŻNIE od K
- **Model 2: rozkład MAAE pretekstu (stopnie)** — CAŁA przewaga K=36 pochodzi z 81× większej liczby par, NIE z rozdzielczości kątowej. NIE mylić z rozkładem RMSE zadania docelowego (§5) — ta sama arytmetyka, inna wielkość i inna jednostka
- **Model 2: transfer RGB2Depth (5 ziaren)** — WYNIK NEGATYWNY — żadna różnica wobec `scratch` nie jest istotna
- **Model 2: transfer pretext_K12_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K36_p16_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K36_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer pretext_K4_seed0 vs scratch** — test Welcha, 5 ziaren; wartość ujemna = lepiej niż scratch
- **Model 2: transfer przy 10% zbioru treningowego** — n_probek_train=4946, 258.8 epok; walidacja i test PEŁNE; podzbiór stratyfikowany po lokalizacji, ziarno 20260815 stałe
- **Model 2: transfer przy 25% zbioru treningowego** — n_probek_train=12366, 103.5 epok; walidacja i test PEŁNE; podzbiór stratyfikowany po lokalizacji, ziarno 20260815 stałe
- **Model 2: pretext_K4_seed0_vs_scratch@10%** — pretrening GORSZY; 0.6-1.9x podlogi szumu
- **Model 2: pretext_K36_seed0_vs_scratch@10%** — pretrening LEPSZY; 0.0-0.0x podlogi szumu
- **Model 2: pretext_K4_seed0_vs_scratch@25%** — pretrening GORSZY; 0.7-2.1x podlogi szumu
- **Model 2: pretext_K36_seed0_vs_scratch@25%** — pretrening GORSZY; 0.6-1.9x podlogi szumu
- **Model 2: pretext_K4_seed0_vs_scratch@100%** — pretrening LEPSZY; 0.4-1.2x podlogi szumu
- **Model 2: pretext_K36_seed0_vs_scratch@100%** — pretrening GORSZY; 0.6-2.0x podlogi szumu
- **Model 2: WERDYKT przewidywania §5.1 — pretext_K4_seed0** — przewidywanie z 2026-08-13 §5.1 WYCOFANE — pretrening nie zaczyna pomagać przy mniejszym zbiorze docelowym
- **Model 2: WERDYKT przewidywania §5.1 — pretext_K36_seed0** — przewidywanie z 2026-08-13 §5.1 WYCOFANE — pretrening nie zaczyna pomagać przy mniejszym zbiorze docelowym
- **echo2depth RMSE test@36, geometria `main` (3 ziarna)** — średnia ± sd po 3 ziarnach
- **echo2depth RMSE test@36, geometria `patched` (3 ziarna)** — średnia ± sd po 3 ziarnach
- **PEŁNY MODEL: RMSE test@36 A/B/D (3 ziarna)** — ZASTĘPUJE wersję z 1 ziarna z 2026-08-13 §2; ziarna 1-2 dołożone POST HOC
- **Pełny model: gestosc_D_minus_A (3 ziarna)** — test SPAROWANY po ziarnie; 2.8-8.9x podlogi szumu
- **Pełny model: ilosc_danych_B_minus_D (3 ziarna)** — test SPAROWANY po ziarnie; 3.9-12.3x podlogi szumu
- **Pełny model: laczny_B_minus_A (3 ziarna)** — test SPAROWANY po ziarnie; 6.7-21.2x podlogi szumu
- **Model 2: MAAE pretekstu wg wariantu — K4** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1; n_ziaren=3, sd=— przy n=1
- **Model 2: MAAE pretekstu wg wariantu — K12** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1; n_ziaren=3, sd=— przy n=1
- **Model 2: MAAE pretekstu wg wariantu — K36** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1; n_ziaren=3, sd=— przy n=1
- **Model 2: MAAE pretekstu wg wariantu — K36@16par** — poziom losowy 90° NIEZALEŻNIE od K — dlatego MAAE, a nie top-1; n_ziaren=3, sd=— przy n=1
- **Model 2: RGB2Depth po pretreningu — pretext_K4_seed0 @ 100% zbioru** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu; 100% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K36_p16_seed0 @ 100% zbioru** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu; 100% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — scratch @ 100% zbioru** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu; 100% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K36_seed0 @ 100% zbioru** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu; 100% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K12_seed0 @ 100% zbioru** — n_ziaren=5; zadanie docelowe BEZ audio w czasie testu; 100% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — scratch_f25 @ 25% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 25% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K36_seed0_f25 @ 25% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 25% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K4_seed0_f25 @ 25% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 25% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K36_seed0_f10 @ 10% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 10% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — scratch_f10 @ 10% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 10% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RGB2Depth po pretreningu — pretext_K4_seed0_f10 @ 10% zbioru** — n_ziaren=3; zadanie docelowe BEZ audio w czasie testu; 10% zbioru TRENINGOWEGO (walidacja i test zawsze pełne)
- **Model 2: RMSE docelowego — SAMA rozdzielczość kątowa** — wartość ujemna = poprawa; to jest rozkład RMSE ZADANIA DOCELOWEGO, NIE rozkład MAAE pretekstu (§4, w stopniach)
- **Model 2: RMSE docelowego — SAMA liczba par** — wartość ujemna = poprawa; to jest rozkład RMSE ZADANIA DOCELOWEGO, NIE rozkład MAAE pretekstu (§4, w stopniach)
- **Model 2: RMSE docelowego — efekt łączny K36 − K4** — wartość ujemna = poprawa; to jest rozkład RMSE ZADANIA DOCELOWEGO, NIE rozkład MAAE pretekstu (§4, w stopniach)

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
| Krzywa nasycenia na NATURALNEJ liczności (C6/C9/C12/C18) | grupa `krzywa`, 12 przebiegów, 10,4 h — SKREŚLONA świadomie | rośnie po gęstości I rozmiarze zbioru naraz; `krzywa_staly` odpowiada na to samo pytanie ostrzej i JEST policzona — luki w tekście NIE zostawiać |
| Rozrzut po ziarnach dla MAAE zadania pretekstowego (§4) | pretext_K{4,12,36} × ziarna 1–2, ~1,6 h | MAAE 61,23 / 55,73 / 25,13 mają n=1 — podane BEZ oszacowania rozrzutu; pominięte świadomie (niski zwrot), ale w tekście musi to być napisane |
| Wariant `patched` na PEŁNYM modelu (PA/PB/PD) | grupa `geometria` — SKREŚLONA świadomie | wada geometrii jest akustyczna, więc `geometria_echo` bada ją ostrzej i ~20× taniej |
| Warunek `ESA` (echo2depth + permutacja kątów w obrębie lokalizacji) | niezaimplementowany — decyzja świadoma | rozdzieliłby 'echo niesie pozycję' od 'echo niesie orientację'; warunek `ESE` odpowiada na słabszą wersję tego pytania i jest policzony |
| Transfer geometrii na office_4 | dowolny warunek `patched` + `main` | sonda przy danych testowych trzymanych dosłownie stałych |

---

## Kontrola spójności: pozycje o tej samej nazwie

Brak — każda pozycja ma unikalną nazwę, więc żadnej liczby nie da się przepisać
z niewłaściwego źródła. Kontrola: `thesis_numbers._duplikaty_nazw()`.
