# Raport sesji — 2026-08-12: uporządkowanie repozytorium, Model 2, kolejka nocna

Sesja dzienna, bez długich przebiegów GPU. Trzy cele: uczynić `my-operations/ml/` czytelnym,
dokończyć kod Modelu 2, przygotować skrypt kolejkujący do uruchomienia wieczorem.

Statusy: **[Z]** zmierzone · **[Z-]** z zastrzeżeniem · **[W]** wywnioskowane z kodu ·
**[X]** nie sprawdzone.

---

## 1. Uporządkowanie `my-operations/ml/` **[Z]**

### 1.1 Inwentaryzacja

24 pliki `.py`, **7 257 linii**, 10 punktów wejścia. **Modułów martwych: 0** — każdy plik jest albo
punktem wejścia, albo importowany przez co najmniej jeden inny. Nic nie było do skasowania.

Pełna mapa (plik → rola → kto importuje) jest w `my-operations/ml/README.md`.

### 1.2 Nowa struktura

Przeniesienia przez `git mv` tam, gdzie plik był w gicie; pozostałe zwykłym `mv` (część plików
z ostatnich dwóch sesji nie była jeszcze zacommitowana — historia i tak zaczyna się przy najbliższym
commicie).

```
ml/  paths.py                      <- korzeń: jedyne źródło ścieżek
     dataset/       angles, echo_h5_dataset, splits, verify_loader, echo_data
     depth_model/   train_condition, evaluate, metrics, fast_bilinear
     pretext_model/ pairs, model, metrics, train_pretext, transfer, summarize
     matrix/        experiments, exp_ctl, ml_ctl
     analysis/      geometry_check, thesis_numbers
     checks/        determinism_check, bench
```

**Nazwy plików `.py` bez zmian** — są cytowane w trzech raportach i w `LICZBY_DO_PRACY.md`.
**Ścieżki w `outputs/ml/` bez zmian** — sprawdzone, `.gitignore` nie odwołuje się do żadnej starej
ścieżki kodu.

Decyzje wymagające uzasadnienia:

- **`paths.py` zostaje w korzeniu.** Importuje go 16 modułów ze wszystkich podpakietów, więc nie
  należy do żadnej kategorii — jest korzeniem drzewa, nie jego gałęzią. Umieszczenie go w `dataset/`
  albo `matrix/` byłoby arbitralne.
- **`bench.py` → `checks/`, nie `dataset/`.** Mierzy przepustowość, czyli jest kontrolą, mimo że
  dotyczy dataloadera. Importuje go `dataset/echo_data.py` — import między podpakietami jest
  poprawny i nie jest problemem.
- **Bootstrap zostaje w `depth_model/metrics.py`**, mimo że jest analizą bez GPU. Wydzielenie
  wymagałoby podziału pliku, a `metrics.py` jest cytowany w raportach.
- **Kolizji nazwa-pakietu/nazwa-modułu nie było** — żadna ścieżka typu `matrix/matrix.py` nie
  powstała, więc żadnego katalogu nie trzeba było przemianowywać.
- **Katalogu `misc/` ani `utils/` nie utworzono** (zgodnie z zakazem) i nie było takiej potrzeby.

### 1.3 Weryfikacja po przenosinach **[Z]**

| kontrola | wynik |
|---|---|
| import wszystkich modułów | **22/22** |
| punkty wejścia odpowiadające na `--help` | **10/10** |
| przebieg dymny `train_condition.py --steps 20` (warunek `EA`) | **przeszedł** end-to-end |
| ścieżki `outputs/ml/` | niezmienione |
| `.gitignore` wobec starych ścieżek | brak odwołań |

**Znaleziony i naprawiony błąd zastany:** `dataset/echo_data.py --help` wywalał się na
`ValueError: unsupported format character 'p'` — niezescapowany `%` w tekście pomocy (argparse
traktuje go jako początek podstawienia). Błąd istniał przed przenosinami i blokował `--help`;
naprawa to jeden znak (`%` → `%%`).

Brief prosił o przebieg dymny z `--max-steps 20`; realna flaga nazywa się **`--steps`** (jest
cytowana w raportach, więc jej nie przemianowałem).

### 1.4 `README.md` — produkt tego bloku

`my-operations/ml/README.md`: mapa w tabelach per podkatalog · komendy do najczęstszych zadań ·
układ `outputs/ml/` z podziałem na pliki dowodowe i duże artefakty · **słownik nazw warunków**
(pozwala odczytać `EPD` bez sięgania do raportu) · kolejność wykonania od zera · lista zasad, których
nie wolno złamać.

---

## 2. Model 2 — dokończenie **[Z]**

Stan zastany okazał się pełniejszy, niż zakładał brief: `metrics.py` **już zawierał** komplet
z §2.1 (MAAE, tolerancje ±10/30/45°, trafność dosłowna, macierz pomyłek, rozbicie ≤20° / >20°),
a `transfer.py` i `summarize.py` istniały i działały. Dołożone zostały brakujące elementy:

### 2.1 Metryki — bez zmian, zweryfikowane

MAAE jako metryka podstawowa (poziom losowy **90° niezależnie od K** — wspólny punkt odniesienia,
podczas gdy top-1 spada z 25 % przy K=4 do 2,8 % przy K=36). Rozbicie ≤20° / >20° jako test
wykonalności najdrobniejszej granulacji jest zaimplementowane i raportowane.

### 2.2 `transfer.py`

Działa, wczytuje wagi enkodera `RGBDepthNet` z checkpointu pretreningu (**35 z 35 kluczy**,
sprawdzone w poprzedniej sesji), resztę inicjalizuje losowo, ten sam budżet i optymalizator co
`train_condition.py`, audio nieużywane w czasie testu. **5 ziaren** jest realizowane przez kolejkę
(`ml_ctl.TRANSFER_SEEDS`), nie przez flagę w skrypcie — jeden przebieg na wywołanie, zgodnie
z konwencją całej fazy ML.

`load_pretrained_encoder()` **przerywa z błędem**, jeśli nie dopasuje żadnego klucza — ciche
`strict=False` dałoby wynik nieodróżnialny od `Scratch` i tabela pięciu warunków byłaby tabelą
pięciu razy tego samego.

### 2.3 `summarize.py` — rozszerzony

Dołożone: **macierze pomyłek** wszystkich wariantów K (trzymane w JSON, nie drukowane — przy K=36
to 1 296 liczb) oraz **rozkład efektu pretreningu**:

```
K36 − K36@16par   izoluje ILOŚĆ PAR
K36@16par − K4    izoluje SAMĄ ROZDZIELCZOŚĆ KĄTOWĄ zadania
```

To ta sama logika, co warunek `D` w Modelu 1: K=36 ma 1 296 par na lokalizację wobec 16 przy K=4,
czyli **81× więcej**, więc porównanie K=4 vs K=36 bez tej kontroli mieszałoby rozdzielczość
z rozmiarem zbioru.

**Wpięcie w `thesis_numbers.py`**: liczby Modelu 2 (MAAE per K, RMSE transferu per inicjalizacja,
rozkład efektu) trafiają do `LICZBY_DO_PRACY.md` automatycznie, gdy tylko `pretext/summary.json`
się pojawi. Sprawdzone na pustym stanie — nie wywala się, po prostu nic nie dodaje.

### 2.4 Asercja protokołu w `evaluate.py --compare` **[Z]**

Z §7 poprzedniego raportu. `_assert_same_protocol()` czyta `status.json` obu przebiegów i porównuje
**`val_angle_subset`** oraz **`mask_mode`**. Przy niezgodności przerywa z komunikatem wskazującym
konkretną różnicę; `--force-compare` pozwala wymusić, ale drukuje ostrzeżenie.

Sprawdzone w obie strony: przebiegi zgodne przechodzą, po podmienieniu `val_angle_subset` na
`cardinal` porównanie zostaje przerwane, a z `--force-compare` przechodzi z ostrzeżeniem.

Powód: na dysku leżą checkpointy sprzed i po zmianie protokołu z 2026-08-11 §1. Checkpoint wybrany
innym kryterium to **inny model** — różnica wyglądałaby jak efekt warunku, a byłaby efektem zmiany
protokołu.

---

## 3. `matrix/ml_ctl.py` — kolejka nocna **[Z] (kod) / [X] (nieuruchomiona)**

47 kroków, **9,79 h GPU, 19,2 GB**. Uruchamiany przez autora wieczorem — **w tej sesji nie był
uruchomiony**.

| # | grupa | kroków | h | GB |
|---|---|---|---|---|
| 1 | Model 2: pretrening (K = 4/12/36 + K36@16par) | 4 | 1,12 | 0,67 |
| 2 | Model 2: transfer (5 warunków × 5 ziaren) + `summarize` | 26 | 5,00 | 4,17 |
| 3 | `EPA`/`EPB`/`EPD` — zamyka maskę ścisłą i sondę `office_4` | 3 | 0,39 | 0,50 |
| 4 | `glowne` (`A`, `D`), 1 ziarno — `B` już policzone | 2 | 1,72 | 11,81 |
| 5 | `krzywa_staly` (`EK6/EK9/EK12/EK18` × 3 ziarna) | 12 | 1,56 | 2,00 |

Model 2 idzie **pierwszy**, bo jest najdłuższy i jako jedyny może jeszcze nie wyjść — ma dostać całą
noc, a nie resztki.

### 3.1 Czasy zmierzone, nie zgadywane **[Z]**

Pierwsza wersja planu miała czasy Modelu 2 oszacowane wzorem — dla planu, który ma pozwolić
zaplanować noc, to za mało. Zmierzone na tym sprzęcie (batch 32, 120 kroków):

| | przepustowość | 40 000 kroków (z 15 % zapasem na walidację) |
|---|---|---|
| pretrening K=4 | 1 651 par/s | 0,25 h |
| pretrening K=36 | 1 384 par/s | 0,30 h |
| transfer | 2 019 próbek/s | 0,20 h |

### 3.2 Co przejęte z `echo_ctl.py`

Plik kolejki JSON (`pending`/`done`/`failed`/`current`, przeżywa restart) · wykrywanie działającego
przebiegu **skanem `/proc`** po wierszu polecenia, nie plikiem PID · log per krok + zbiorcze
podsumowanie · znaczniki czasu w każdej linii nadzorcy · `plan` pokazujący pełny zamiar bez
uruchamiania · `start_new_session=True`, żeby zerwane SSH nie zabiło przebiegu · źródłem prawdy
o „gotowe" jest **artefakt** (`status.json: finished`), nie własna księgowość kolejki.

### 3.3 Jedna świadoma różnica wobec `echo_ctl.py`

Tam druga nieudana próba **przerywa całą kolejkę** — słusznie, bo typowa przyczyna to zawieszony
GPU. Tutaj krok, który padł, jest logowany i **kolejka idzie dalej**: kroki 3–5 nie zależą od
Modelu 2, więc jego niepowodzenie nie może skasować reszty nocy. Zabezpieczeniem przed „dopisywaniem
błędów przez całą noc" jest kontrola wolnego miejsca i podsumowanie z kodami wyjścia.

Dodatkowo: krok transferu **wymagający enkodera z pretreningu** jest jawnie **pomijany**, gdy tamten
plik nie powstał — z wpisem w podsumowaniu, zamiast startować i wywalać się.

### 3.4 Czego z `echo_ctl.py` NIE przeniesiono

- **interaktywne menu** (`watch`, klawisze `s`/`d`/`q`) — kolejka ma chodzić bez człowieka przy
  klawiaturze; stan pokazuje `status`;
- **`verify`** (kompletność HDF5) — to własność zbioru danych, sprawdzana przez
  `dataset/echo_data.py --verify-loader`, nie przez trening;
- **ponawianie kroku** — przebieg treningowy wznawia się z `--resume` sam, a ponawianie w pętli
  maskowałoby prawdziwą przyczynę awarii.

### 3.5 Pozostałe wymagania

Wznawianie (kroki z `finished=true` pomijane bez liczenia) · kontrola wolnego miejsca **przed
każdym krokiem**, próg **15 GB** · `thesis_numbers.py` **po każdym kroku**, żeby
`LICZBY_DO_PRACY.md` był aktualny nad ranem niezależnie od tego, dokąd kolejka dojdzie ·
ewaluacja (`evaluate.py`) od razu po każdym treningu Modelu 1 · logi w `outputs/ml/logs/` +
podsumowanie `ml_ctl_<data>.md` z czasami i wolnym miejscem na końcu.

---

## 3.6 `exp_ctl.py` — przegląd po pytaniu autora: nie jest przestarzały, ale **kłamał** [Z]

Pytanie brzmiało, czy `exp_ctl.py` nie jest przestarzały i czy nie podaje danych częściowo
wyrenderowanych. **Nie jest przestarzały** (jego `status`/`plan`/`results` nie mają odpowiednika
w `ml_ctl.py`, który jest kolejką, nie pulpitem), ale **podawał złe liczby**. Trzy defekty, wszystkie
naprawione.

### Defekt 1: martwy odczyt `metrics.jsonl` [Z]

`exp_ctl` czytał `rekord["overall"]["RMSE"]`. Po zmianie `evaluate()` z 2026-08-11 §1 rekordy mają
klucz **`all`**, nie `overall` — `KeyError` był po cichu połykany przez `except`. Skutek: dla
**działającego** przebiegu kolumna `best RMSE` pokazywała `-`, mimo że wynik był już w logu.
Naprawione czytnikiem `_block()`, który zna oba formaty (na dysku leżą pliki obu).

### Defekt 2: `metrics.jsonl` zawiera DWA przebiegi naraz — przyczyna źródłowa [Z]

`train_condition.py` otwiera `metrics.jsonl` i `train_loss.csv` w trybie **dopisywania**, a `--force`
nie kasuje katalogu. Ponowne uruchomienie warunku zostawiało więc w logu wpisy **starego i nowego**
przebiegu. Zmierzone:

| przebieg | rekordów | stary format | nowy format |
|---|---|---|---|
| `EA_seed0`, `EB_seed0`, `ED_seed0`, `ESE_seed0` | **80** | 40 | 40 |
| pozostałe 10 | 40 | 0 | 40 |

Zanieczyszczone są dokładnie te cztery, które w §3 poprzedniej sesji ponowiono z `--force`.
Ponieważ `exp_ctl` brał **minimum po całym pliku**, dla `EA_seed0` pokazywał **0,56353** zamiast
prawdziwych **0,76085** — czyli wynik przebiegu **sprzed** zmiany protokołu, który walidował na
4 kątach i z tego powodu miał niższe RMSE. Liczba zaniżona o 26 %, bez żadnego ostrzeżenia.

Naprawione **w przyczynie**: `train_condition.py` kasuje oba logi na świeżym starcie (przy
`--resume` nie kasuje — tam ciągniemy ten sam przebieg). Naprawione też **w czytniku**:
`status.json` jest teraz źródłem prawdy, a `metrics.jsonl` uzupełnia wyłącznie wtedy, gdy statusu
jeszcze nie ma (przebieg w toku) — stare katalogi na dysku nadal bywają zanieczyszczone.

**Żadna liczba w raportach nie była tym dotknięta**: wszystkie pochodzą z `evaluate.py` na zbiorze
testowym albo bezpośrednio ze `status.json`, nie z `exp_ctl`.

### Defekt 3: kolumna bez etykiety, protokół niewidoczny [Z]

`best RMSE` to **walidacyjne** RMSE, nie testowe — stąd 0,76085 obok 0,77895 z `test@36` dla tego
samego przebiegu. Bez etykiety łatwo zestawić ze sobą dwie różne wielkości. Dodane: nazwa kolumny
`best val RMSE`, przypis pod tabelą, oraz kolumna **`prot`** pokazująca protokół walidacji
(`36` = po zmianie z §1, czerwone `STARY` = sprzed) z podsumowaniem, ile przebiegów jest
nieporównywalnych.

### Czy zaśmieca dysk

**Nie.** `exp_ctl.py` zapisuje wyłącznie `outputs/ml/experiments.json` (zamierzone, przy `plan`)
oraz `outputs/ml/results_summary.json` przy podkomendzie `results` — ta ostatnia nigdy nie była
uruchamiana, pliku nie ma.

### Co zostaje do decyzji autora

Cztery pliki `metrics.jsonl` zawierają po 40 nieaktualnych rekordów sprzed zmiany protokołu.
**Po naprawie czytnika są nieszkodliwe** — nic ich już nie czyta jako źródła prawdy. Można je
przyciąć, ale to kasowanie danych pomiarowych, więc **nie zostało zrobione samodzielnie**. Stare
wyniki są i tak zachowane w `RAPORT_SESJI_2026-08-10.md` §5.6 oraz
`outputs/ml/echo_ablation/echo_density_seed0.json`.

### 3.7 Kolumna `plan` w `exp_ctl status` — i dwa błędy znalezione przy okazji [Z]

`exp_ctl` pokazywał **„gotowych 14/66"**, co czyta się jako 21 % postępu. Mianownik 66 to
**przestrzeń projektowa** (22 warunki × 3 ziarna), a nie plan: 35 z nich zostało po drodze
odwołanych albo odsuniętych. Wobec faktycznego planu jest to **14/31, czyli 45 %**.

Powody trafiły do `experiments.py` (`PLANNED_SEEDS`, `DEFERRED_GROUPS`, `plan_status()`) — bo są
decyzjami o macierzy, a nie właściwością widoku. `exp_ctl` je tylko wyświetla: kolumna `plan`
(`tak` / `—`) plus rozbicie pod tabelą, kto i dlaczego wypadł.

| n | co | powód |
|---|---|---|
| 12 | `C6/C9/C12/C18` × 3 | krzywa rośnie po gęstości **i** rozmiarze zbioru naraz (10.08 §5.1); zastąpiona przez `krzywa_staly`; najdroższa grupa (10,4 h, 70,8 GB) |
| 9 | `PA/PB/PD` × 3 | wada geometrii jest **akustyczna** (+46,3 % energii późnej wobec +1,3 % całkowitej) → `geometria_echo` bada to ostrzej i ~20× taniej |
| 6 | `A/B/D` ziarna 1–2 | **degradacja 11.08 §2**: `bound` = 0,00529 < 0,015 |
| 6 | `EPA/EPB/EPD` ziarna 1–2 | do domknięcia maski ścisłej wystarczy 1 model `patched` |
| 2 | `SE` ziarna 1–2 | bramka, nie pozycja w tabeli pracy |

Rozróżnienie utrzymane w kodzie: **6 przebiegów jest odwołanych regułą** (nie wrócą bez nowego
pomiaru), pozostałe **29 jest odsuniętych** — warunki są zdefiniowane i gotowe, wracają, jeśli
wyniki tego zażądają.

**Dwa błędy znalezione przy tej okazji:**

1. **`TRAIN_SCRIPT` wskazywał nieistniejącą ścieżkę.** Było
   `Path(__file__).parent / "train_condition.py"`, czyli po reorganizacji z §1 —
   `matrix/train_condition.py`. Plik mieszka w `depth_model/`. **`exp_ctl start` był zepsuty**,
   a `next` proponował komendy, które by się nie uruchomiły. Kontrola `--help` z §1.3 tego nie
   złapała, bo ścieżka jest liczona przy imporcie, ale używana dopiero przy `start`/`next`.
   Naprawione, z asercją istnienia pliku przy imporcie — żeby następna reorganizacja wywaliła się
   od razu, a nie po cichu.
2. **`next` proponował przebiegi spoza planu** — pierwszym podpowiadanym był `SE --seed 1`,
   odwołany decyzją z 11.08 §2. Czyli narzędzie podpowiadało robotę, o której zapadła decyzja,
   żeby jej nie robić. Teraz filtruje po `plan_status()`.

Ścieżki w `ml_ctl.py` sprawdzone osobno — wszystkie 6 wskazuje istniejące pliki.

---

## 4. NYU-V2 i DIODE — decyzja odłożona z uzasadnieniem **[W]**

Gao testuje transfer reprezentacji na **NYU-V2** i **DIODE** — zbiorach rzeczywistych, bez echa.
Pytanie „czemu ich u nas nie ma" musi trafić do rozdziału o ograniczeniach, a nie zostać
przemilczane. Odpowiedź jest **różna dla obu modeli**.

**Model 1 — niewykonalne.** Wymaga echa na wejściu w czasie testu. NYU-V2 i DIODE nie mają danych
akustycznych i nie da się ich wyrenderować: to skany rzeczywiste bez geometrii nadającej się do
symulacji akustycznej (brak zamkniętych, wodoszczelnych siatek i przypisania materiałów, których
wymaga RLRAudioPropagation). Nie jest to kwestia budżetu, tylko braku wejścia.

**Model 2 — wykonalne i naukowo atrakcyjne.** Transfer używa **wyłącznie enkodera wizualnego**, bez
audio w czasie testu — dokładnie ten układ, którego użył Gao. Byłby to test, czy reprezentacja
wyuczona na echach **z symulacji** przenosi się na **obrazy rzeczywiste**, co jest twierdzeniem
mocniejszym niż transfer wewnątrz Repliki.

### Szacunek kosztu (nie implementowane)

| pozycja | szacunek | uwagi |
|---|---|---|
| NYU-V2 Depth V2, podzbiór z etykietami | **~2,8 GB** (1 449 par RGB-D) | wersja `labeled`, `.mat`; pełny surowy zbiór to ~428 GB i nie jest potrzebny |
| DIODE Indoor (val+train) | **~20–80 GB** | znacznie droższy; **rozważyć pominięcie**, NYU-V2 wystarcza do postawienia twierdzenia |
| czas transferu | **~0,2 h na przebieg** | rozmiar zbioru mniejszy niż nasze 49 464 próbki; przy 5 warunkach × 3 ziarna ≈ **3 h** |
| nowy dataloader | **~150–200 linii** | inny format (`.mat` / PNG+NPY), inne `max_depth` (NYU-V2: 10 m), inna rozdzielczość (640×480 → 128×128) |
| enkoder ResNet-50 | **zmiana architektury** | Gao używa U-Netu **tylko dla Repliki**; dla NYU-V2 i DIODE stosuje ResNet-50. Trzymanie się jego układu wymaga drugiej ścieżki pretreningu, bo enkoder pretekstowy musi pasować do zadania docelowego |

**Warunek wejścia:** rozszerzenie ma sens **wyłącznie wtedy, gdy Model 2 pokaże efekt na Replice**.
Jeśli pretrening orientacyjny nie poprawia RGB2Depth na danych, z których pochodzi, testowanie
transferu na innej domenie nie odpowie na żadne pytanie.

**Największy koszt ukryty** to nie dane ani GPU, lecz **ResNet-50**: dołożenie drugiego enkodera
oznacza drugą ścieżkę pretreningu i podwojenie macierzy Modelu 2. Wersja minimalna — NYU-V2 z tym
samym enkoderem U-Net, świadomie odbiegając od Gao — kosztuje ~3 h GPU i ~200 linii, ale wymaga
jawnego zastrzeżenia, że architektura enkodera jest inna niż w pracy odniesienia.

---

## 5. Czego **NIE** zrobiono **[X]**

- **Kolejka nie została uruchomiona** — zgodnie z poleceniem. Żaden przebieg GPU poza pomiarami
  przepustowości (120 kroków) i przebiegiem dymnym (20 kroków), oba skasowane.
- **NYU-V2 / DIODE nie zaimplementowane** — sam szacunek (§4).
- **Nie zweryfikowano `ml_ctl.py` w boju.** Sprawdzone `plan`, `status`, `--help` i logika
  pomijania kroków gotowych; **ścieżka `run` nie została wykonana ani razu**. Ryzyko resztkowe:
  literówka w argumentach któregoś podprocesu ujawni się dopiero przy pierwszym kroku. Zalecenie:
  uruchomić wieczorem i **sprawdzić pierwszy krok po ~2 minutach**, zanim autor pójdzie spać.
- **Nie zmierzono czasu ewaluacji doklejanej po każdym treningu** — dla `echo2depth` to ~4 s,
  dla pełnego modelu ~10 s, więc pomijalne, ale w planie nieuwzględnione.
- **`exp_ctl.py` i `ml_ctl.py` częściowo się pokrywają** (oba potrafią uruchomić warunek Modelu 1).
  Zauważone, nie scalone — `exp_ctl.py` jest pulpitem (`status`/`plan`/`results`), `ml_ctl.py`
  kolejką; scalanie byłoby refaktorem poza zakresem.
- **Nie przycięto czterech zanieczyszczonych `metrics.jsonl`** (§3.6) — po naprawie czytnika są
  nieszkodliwe, a przycięcie kasuje dane pomiarowe, więc czeka na decyzję autora.
- **Historia gita dla plików przeniesionych zwykłym `mv`** (te z ostatnich dwóch sesji, jeszcze
  niezacommitowane) zaczyna się dopiero przy najbliższym commicie. Dla plików już śledzonych
  `git mv` zachował historię.
- **`README.md` nie jest testowany automatycznie** — komendy w nim są przepisane ręcznie i mogą się
  rozjechać z kodem przy kolejnej zmianie interfejsu.

## 6. Do zrobienia w następnej sesji

1. **Uruchomić kolejkę** (`ml_ctl.py run`) i rano przejrzeć `outputs/ml/logs/ml_ctl_<data>.md`.
2. Jeśli Model 2 pokaże efekt — **decyzja o NYU-V2** na podstawie §4.
3. Po zakończeniu kolejki: `analysis/thesis_numbers.py` (kolejka robi to sama po każdym kroku, ale
   warto sprawdzić listę „liczb, których jeszcze nie ma" — powinna się skrócić o pozycje Modelu 2).
