# `my-operations/ml/` — mapa fazy uczenia maszynowego

Kod fazy ML pracy „Visual Echoes 2.0". Napisany **obok** `beyond-image-to-depth/` (repozytorium
Paridy), którego **nie wolno modyfikować** — zmienną niezależną pracy jest gęstość kątowa, więc
sieć, strata i optymalizator muszą zostać dokładnie te opublikowane.

Nazwy katalogów i plików są **po angielsku**, komentarze i dokumentacja **po polsku**.

---

## 1. Mapa: co za co odpowiada

### korzeń

| plik | co robi | rodzaj |
|---|---|---|
| `paths.py` | jedyne miejsce, z którego wyprowadzane są wszystkie ścieżki | biblioteka |

`paths.py` **celowo zostaje w korzeniu**: importuje go 16 modułów ze wszystkich podpakietów, więc
nie należy do żadnej kategorii — jest korzeniem, nie składnikiem.

### `dataset/` — wczytywanie danych

| plik | co robi | rodzaj |
|---|---|---|
| `angles.py` | filtr orientacji: `all`, `cardinal`, `every_N`, `random_K`, `random_K_of_G` — **zmienna niezależna pracy** | biblioteka |
| `echo_h5_dataset.py` | `EchoH5Dataset`: czyta echo/RGB/głębię z HDF5, maski przecięcia i ścisła, permutacja echa | biblioteka |
| `splits.py` | podział train/val/test **po lokalizacjach**, deterministyczny, zapisywany do JSON | biblioteka |
| `verify_loader.py` | 8 testów poprawności dataloadera (rozłączność, liczności, NaN, zakresy) | biblioteka |
| `echo_data.py` | CLI strony danych: `--verify-loader`, `--bench`, budowa podziału | **punkt wejścia** |

### `depth_model/` — Model 1 (Parida: RGB + echo → głębia)

| plik | co robi | rodzaj |
|---|---|---|
| `train_condition.py` | uruchamia **jeden warunek przy jednym ziarnie**; walidacja na 36 kątach, dwa checkpointy | **punkt wejścia** |
| `evaluate.py` | protokół ewaluacji: `test@36`/`test@4`, krzywa kątowa, per scena, bootstrap sparowany | **punkt wejścia** |
| `metrics.py` | metryki zgodne z Paridą + statystyki per próbka + bootstrap po lokalizacjach | biblioteka |
| `fast_bilinear.py` | tożsamy matematycznie, 16× szybszy zamiennik `nn.Bilinear` | biblioteka |

Bootstrap siedzi w `metrics.py`, mimo że jest analizą bez GPU — rozdzielenie wymagałoby podziału
pliku, a `metrics.py` jest cytowany w trzech raportach.

### `pretext_model/` — Model 2 (zadanie orientacyjne)

| plik | co robi | rodzaj |
|---|---|---|
| `pairs.py` | pary (widok z orientacji *i*, echo z orientacji *j*) z tej samej lokalizacji | biblioteka |
| `model.py` | sieć pretekstowa: enkoder U-Net + Echo-Net + fuzja + głowa K klas | biblioteka |
| `metrics.py` | MAAE, tolerancje ±10/30/45°, macierz pomyłek, rozbicie ≤20° / >20° | biblioteka |
| `train_pretext.py` | pretrening zadania orientacyjnego dla danego K | **punkt wejścia** |
| `transfer.py` | zadanie docelowe RGB2Depth **bez audio**, enkoder z pretreningu | **punkt wejścia** |
| `summarize.py` | składa tabele: MAAE per K, transfer, rozkład efektu | **punkt wejścia** |

### `matrix/` — macierz eksperymentów

| plik | co robi | rodzaj |
|---|---|---|
| `experiments.py` | definicja 22 warunków, budżet czasu i dysku, zmierzone liczby parametrów | biblioteka |
| `exp_ctl.py` | pulpit macierzy Modelu 1: `plan`, `status`, `next`, `start`, `stop`, `results` | **punkt wejścia** |
| `ml_ctl.py` | **kolejka nocna**: cała sekwencja Model 2 → geometria → główne → krzywa | **punkt wejścia** |

### `analysis/` — bez GPU

| plik | co robi | rodzaj |
|---|---|---|
| `geometry_check.py` | rozstrzygnięcie `main` vs `patched`: kanały, energia, kontrast kątowy | **punkt wejścia** |
| `thesis_numbers.py` | eksport **każdej zmierzonej liczby** do `docs/LICZBY_DO_PRACY.md` | **punkt wejścia** |

### `checks/` — kontrole jednorazowe

| plik | co robi | rodzaj |
|---|---|---|
| `determinism_check.py` | podłoga szumu frameworka i decyzja o `--fast-bilinear` | **punkt wejścia** |
| `bench.py` | przepustowość dataloadera i modelu | biblioteka |

---

## 2. Jak uruchomić najczęstsze rzeczy

```bash
conda activate habitat

# plan macierzy: co, ile godzin, ile GB -- nic nie uruchamia
python my-operations/ml/matrix/exp_ctl.py plan --fast-bilinear

# jeden przebieg (jeden warunek, jedno ziarno)
python my-operations/ml/depth_model/train_condition.py --condition EA --seed 0 --fast-bilinear

# ewaluacja przebiegu: test@36 i test@4, krzywa kątowa, per scena
python my-operations/ml/depth_model/evaluate.py --run-dir outputs/ml/runs/EA_seed0

# porównanie dwóch warunków: różnica + przedział ufności (bootstrap po lokalizacjach)
python my-operations/ml/depth_model/evaluate.py --compare EA_seed0 EB_seed0

# odświeżenie LICZB_DO_PRACY.md po każdym nowym wyniku
python my-operations/ml/analysis/thesis_numbers.py

# kolejka nocna: najpierw zobacz plan, dopiero potem uruchom
python my-operations/ml/matrix/ml_ctl.py plan
python my-operations/ml/matrix/ml_ctl.py run
```

Sprawdzenie strony danych po zmianach w `dataset/`:

```bash
python my-operations/ml/dataset/echo_data.py --verify-loader --geometry main
```

---

## 3. Gdzie co ląduje — `outputs/ml/`

**Ścieżki w `outputs/ml/` są cytowane w raportach i w `LICZBY_DO_PRACY.md` — nie zmieniać.**

| katalog | zawartość | w gicie? |
|---|---|---|
| `splits/` | podział lokalizacji + odcisk `e0bf7547668d9e0a` | **tak** |
| `experiments.json` | konfiguracja macierzy + budżet | **tak** |
| `geometry_check/` | rozstrzygnięcie `main` vs `patched` | **tak** |
| `determinism/` | podłoga szumu, decyzja `--fast-bilinear` | **tak** |
| `echo_ablation/` | wkład echa, rozkład efektu, tabela luki, 3 ziarna | **tak** |
| `eval/<run>/eval.json` | metryki przebiegu | **tak** |
| `eval/compare_*.json` | różnice z przedziałami ufności | **tak** |
| `mask_check/`, `disk_budget.json`, `thesis_numbers.json` | kontrole i eksport | **tak** |
| `verify_loader/` | raport 8 testów dataloadera | **tak** |
| `eval/<run>/samples_*.npz` | tabele per próbka | nie (duże) |
| `runs/`, `pretext/`, `pretext_transfer/` | checkpointy, ~5,9 GB na przebieg pełnego modelu | nie |
| `logs/` | logi kolejki `ml_ctl.py` | nie |

Reguła: **małe pliki JSON, na których stoją liczby w pracy — tak; wszystko duże i odtwarzalne — nie.**

---

## 4. Słownik nazw warunków

Identyfikator warunku czyta się z lewej do prawej: **przedrostki** + **rdzeń**.

| element | znaczenie |
|---|---|
| `A` | `cardinal` — 4 kierunki kardynalne, baseline VisualEchoes (Gao 2020) |
| `B` | `all` — wszystkie 36 orientacji |
| `D` | `random_4` — 4 kąty **losowane per lokalizacja**, liczność równa `A` |
| `C6`/`C9`/`C12`/`C18` | krzywa nasycenia: 6/9/12/18 orientacji, liczność naturalna |
| `E…` | **`echo2depth`** — sama gałąź audio, bez obrazu (np. `EA`, `EB`, `ED`) |
| `P…` | geometria **`patched`** — sceny z domkniętym sufitem (np. `PA`, `PB`, `PD`) |
| `EP…` | oba naraz: `echo2depth` **i** `patched` (np. `EPD`) |
| `SE` | pełny model, **echo permutowane** — kontrola „ile w ogóle wnosi echo" |
| `ESE` | to samo na `echo2depth` |
| `EK6`…`EK18` | krzywa przy **stałym budżecie**: 4 próbki/lokalizację losowane z siatki K orientacji |

Czyli **`EPD`** = `echo2depth` + `patched` + `random_4`.

Grupy (`experiments.GROUPS`, w kolejności uruchamiania): `bramka` → `echo` → `glowne` → `krzywa`
→ `krzywa_staly` → `geometria_echo` → `geometria`.

---

## 5. Kolejność, jeśli zaczynasz od zera

1. **Podział** — `dataset/echo_data.py` zbuduje `outputs/ml/splits/replica_locations.json`
   (odcisk musi wyjść `e0bf7547668d9e0a`; jeśli inny, coś się zmieniło w danych).
2. **Weryfikacja dataloadera** — `echo_data.py --verify-loader --geometry {main,patched}`, 8/8 PASS.
3. **Kontrole** — `analysis/geometry_check.py` (geometria) i `checks/determinism_check.py`
   (podłoga szumu). Obie zapisują pliki dowodowe, na których stoją późniejsze interpretacje.
4. **Grupy macierzy** — `matrix/exp_ctl.py plan`, potem `bramka` → `echo` → reszta.
   Albo od razu `matrix/ml_ctl.py run` (kolejka nocna).
5. Po każdym wyniku: `analysis/thesis_numbers.py`.

---

## 6. Zasady, których nie wolno złamać

- `beyond-image-to-depth/` — **zero zmian** (`git diff` musi być pusty).
- Podział lokalizacji jest **zamrożony**: odcisk `e0bf7547668d9e0a`.
- `max_depth = 14,104` m, **40 000 kroków**, **batch 32** — jednakowo we wszystkich warunkach.
- Walidacja **zawsze na 36 kątach**, niezależnie od podzbioru treningowego warunku
  (`depth_model/train_condition.VAL_ANGLE_SUBSET`).
- Kryteriów zapisanych przed pomiarem nie przepisuje się po zobaczeniu danych.
