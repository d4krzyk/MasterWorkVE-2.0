# Raport sesji — 2026-08-05: dataloader i szkielet eksperymentów

Dokument do pracy magisterskiej. Sesja domyka przejście z fazy **generowania danych** do fazy
**uczenia maszynowego**. Żaden model nie był trenowany (poza mikrobenchmarkiem i dwoma
przebiegami dymnymi po 40–60 kroków, usuniętymi po sprawdzeniu) — celem było zbudowanie
i **zweryfikowanie pomiarem** warstwy, która karmi sieć, oraz ustalenie, ile realnie potrwa
macierz eksperymentów.

## Legenda statusów

| status | znaczenie |
|---|---|
| **[Z]** | **Zmierzone** — istnieje skrypt, surowe wyjście, liczby w dokumencie. Nadaje się do pracy jako wynik. |
| **[Z-]** | Zmierzone, ale z zastrzeżeniem, które trzeba cytować razem z liczbą. |
| **[W]** | **Wywnioskowane** z kodu źródłowego, nie z pomiaru. |
| **[X]** | **Nie sprawdzone** — wymieniam, żeby nie powstało wrażenie, że zostało. |

---

## 1. Punkt wyjścia

Faza generowania zamknięta: dataset kompletny w obu wariantach geometrii, 28/28 plików HDF5
z `complete=True`. Baza kodu uczącego to repozytorium Paridy (*Beyond Image to Depth*, CVPR 2021),
podlinkowane jako `beyond-image-to-depth/`.

Ograniczenie narzucone na sesję i utrzymane: **nie modyfikować plików Paridy**. Wszystkie 15 plików
`.py` tego repozytorium jest bit-identycznych ze stanem sprzed sesji; cały nowy kod stoi obok,
w `my-operations/ml/`. Powód jest metodologiczny: zmienną niezależną pracy jest gęstość kątowa,
więc sieć, strata i optymalizator muszą zostać dokładnie te opublikowane — każda zmiana ich kodu
odbiera możliwość przypisania efektu.

---

## 2. Co powstało

`my-operations/ml/`, 3 193 linie w 13 plikach:

| plik | rola |
|---|---|
| `paths.py` | jedno źródło wszystkich ścieżek fazy uczenia (jak `echo_core/paths.py` w generatorze) |
| `splits.py` | podział train/val/test **po lokalizacjach**, deterministyczny, zapisywany do JSON |
| `angles.py` | filtr orientacji — zmienna niezależna pracy |
| `echo_h5_dataset.py` | `EchoH5Dataset` — zamiennik `AudioVisualDataset`, czyta HDF5 leniwie |
| `verify_loader.py` + `echo_data.py` | weryfikacja (8 testów) i CLI strony danych |
| `bench.py` | benchmark przepustowości: dataloader, model, wariant bez kompresji |
| `metrics.py` | metryki standardowe + stratyfikowane + per scena |
| `experiments.py` | definicja macierzy warunków |
| `train_condition.py` | uruchamia jeden warunek przy jednym ziarnie |
| `exp_ctl.py` | pulpit kontrolny macierzy (analogicznie do `echo_ctl.py`) |
| `fast_bilinear.py` | **opcjonalny**, domyślnie wyłączony, szybszy zamiennik `nn.Bilinear` |

Artefakty w `outputs/ml/` (poza gitem): `splits/`, `verify_loader/{main,patched}/`, `bench/`,
`experiments.json`, `runs/`.

---

## 3. Ustalenia twarde

### 3.1 Zapisane spektrogramy są bit-zgodne z potokiem Paridy **[Z]**

Zbiór przechowuje gotowe spektrogramy `(2, 257, 166)` float16, więc `librosa.stft` znika z gorącej
pętli `__getitem__`. Trzeba było jednak wykazać, że to *ten sam* spektrogram, a nie podobny.

`generate_spectrogram()` Paridy woła `librosa.stft(audio, n_fft=512, win_length=64)` **bez** podania
`hop_length`. Sprawdzone w źródle librosy 0.11.0: przy nieokreślonym `hop_length` biblioteka
przyjmuje `win_length // 4`, czyli **16**. Generator (`echo_core/spectrogram.py`) podaje `hop=16`
jawnie. Oba potoki dają więc identyczną siatkę czasową:
`1 + 2646 // 16 = 166` ramek, co zgadza się z `opt.audio_shape = [2,257,166]` z `base_options.py`.

Jedyna różnica to precyzja przechowywania: float16 na dysku wobec float32 w locie. Przy wartościach
rzędu 10 to błąd względny ~5·10⁻⁴.

### 3.2 `scenes['val']` nigdy nie jest ustawiane w kodzie Paridy **[Z]**

`options/base_options.py` dla Repliki definiuje `scenes['train']` (15 scen) i `scenes['test']`
(3 sceny), ale **nie** `scenes['val']` — mimo że `train.py` przełącza `opt.mode = 'val'` i buduje
z niego dataloader. W opublikowanym kodzie ta ścieżka kończy się `KeyError`.

Rozwiązanie przyjęte: 3 sceny held-out (`apartment_2`, `frl_apartment_5`, `office_4`) dzielone
**po lokalizacjach 50/50**, osobno w obrębie każdej sceny.

Dlaczego po lokalizacjach, a nie po próbkach: jedna lokalizacja daje 36 próbek różniących się
wyłącznie orientacją agenta — pozycja jest ta sama. Losowy podział po próbkach wpuściłby tę samą
lokalizację równocześnie do walidacji i do testu, w innych orientacjach. To wyciek: checkpoint
byłby wybierany na podstawie pozycji, na której model jest potem oceniany. **Przy 36 orientacjach
ten wyciek jest 9× groźniejszy niż w oryginalnym układzie 4-kierunkowym Gao.**

Dlaczego osobno w obrębie sceny, a nie na wspólnej puli: losowanie na wspólnej puli mogłoby dać
walidację zdominowaną przez jedną scenę, a wtedy wybór checkpointu premiowałby model dobry akurat
na niej.

Wynik: train 1 374 lokalizacji, val 183, test 183 (razem 1 740). Odcisk podziału
`e0bf7547668d9e0a`; dwa niezależne przebudowania dają ten sam odcisk **[Z]**.

### 3.3 `location_id` i pozycje są bit-identyczne między wariantami geometrii **[Z]**

Sprawdzone dla wszystkich wspólnych scen: `locations/loc_id` identyczne, `locations/position`
różnica maksymalna `0.000000`. Warianty różnią się wyłącznie siatką, nie zbiorem lokalizacji.
Konsekwencja praktyczna: **jeden plik podziału obsługuje oba warianty**, więc porównanie
main-vs-patched idzie na dokładnie tych samych lokalizacjach i nie wymaga osobnego uzasadnienia.

### 3.4 Weryfikacja dataloadera: 8/8 PASS w obu wariantach **[Z]**

`python my-operations/ml/echo_data.py --verify-loader --geometry {main,patched}`

| test | wynik |
|---|---|
| rozłączność train/val/test po lokalizacjach | 0 wspólnych we wszystkich trzech parach; sceny też rozłączne |
| zgodność składu scen z `base_options.py` | tak, pokrycie wszystkich 18 scen |
| liczności 12 podzbiorów kątów | wszystkie co do sztuki (tabela niżej) |
| `random_K` odtwarzalny | to samo ziarno → identyczny indeks; inne ziarno → inny; rozkład kątów w granicach 5σ |
| kształt/dtype wsadu | `img (B,3,128,128)`, `depth (B,1,128,128)`, `audio (B,2,257,166)`, float32 |
| NaN/Inf | 0 |
| echo jest nieujemne (magnituda STFT) | tak |
| zakres głębi vs `max_depth` | patrz 3.5 |

Liczności (razem train+val+test), wariant `main`:

| subset | kątów/lok. | próbek | | subset | kątów/lok. | próbek |
|---|---|---|---|---|---|---|
| `all` | 36 | 62 640 | | `every_3` | 12 | 20 880 |
| `every_2` | 18 | 31 320 | | `every_4` | 9 | 15 660 |
| `every_6` | 6 | 10 440 | | `every_9` / `cardinal` | 4 | 6 960 |

Liczba oczekiwana jest liczona z samego podziału (lokalizacje × kąty na lokalizację), a nie z
drugiego przebiegu tego samego kodu — test naprawdę coś sprawdza, a nie porównuje funkcji z sobą.

Wariant `patched` daje te same liczby, bo składa się z 10 scen załatanych plus 8 szczelnych
z wariantu głównego (GENERATOR_PARAMS.md §4.5) — łącznie znów 18 scen i 1 740 lokalizacji.

### 3.5 Piksele powyżej `max_depth` są ilościowo nieistotne **[Z]**

Pełny skan wszystkich 62 640 próbek, **1 026 293 760 pikseli** (nie próbka — liczba idzie do pracy):

- global max **14,7779 m** wobec `max_depth = 14,104 m`
- pikseli powyżej progu: **131**, czyli **1,3·10⁻⁵ %**
- **wszystkie 131 leży w `apartment_0`** — scenie *treningowej*. Zbiór testowy jest nietknięty.

Znaczenie: model liczy `sigmoid(x) · max_depth`, więc jego wyjście nigdy nie przekroczy 14,104 m
i każdy piksel prawdy powyżej wnosi do RMSE stały, nieusuwalny błąd. Przy tym udziale efekt jest
poniżej progu istotności dla jakiejkolwiek raportowanej metryki.

### 3.6 8,48 % pikseli głębi to zera, rozłożone bardzo nierówno **[Z]**

To ta sama wielkość, która w `train.py` jest maskowana przez `depth_gt != 0`. Rozkład per scena
jest jednak skrajnie nierówny i to ma znaczenie dla interpretacji metryk:

| scena | % zer | | scena | % zer |
|---|---|---|---|---|
| `frl_apartment_1` | 15,17 | | `apartment_1` | 6,05 |
| `frl_apartment_0` | 15,12 | | `office_2` | 6,26 |
| `frl_apartment_3` | 14,83 | | `office_3` | 1,04 |
| `frl_apartment_5` (held-out) | 14,24 | | `apartment_0` | 0,0007 |
| `apartment_2` (held-out) | 11,39 | | `office_4` (held-out) | 0,0012 |

Dwie z trzech scen held-out mają 11–14 % dziur, trzecia praktycznie zero. **Rozbicie metryk per
scena jest z tego powodu obowiązkowe** — uśredniona liczba po zbiorze testowym miesza sceny
o nieporównywalnym pokryciu prawdą.

### 3.7 Dataloader nie jest wąskim gardłem — z ogromnym zapasem **[Z]**

Batch 32, wariant `main`, `angle_subset=all`, augmentacja PIL włączona:

| `num_workers` | próbek/s | wsadów/s |
|---|---|---|
| 0 | 321,6 | 10,05 |
| 2 | 793,3 | 24,79 |
| 4 | 1 603,7 | 50,12 |
| **8** | **2 645,1** | **82,66** |

`prefetch_factor` przy 8 workerach: 2 → 2 652,4; 4 → 2 604,9; 8 → 2 550,0 próbek/s.
**Podnoszenie prefetch szkodzi**, domyślne 2 jest optymalne.

Wariant bez kompresji (jedna scena, `apartment_2`): 4 793,3 wobec 2 873,3 próbek/s, czyli
**1,67× szybciej** przy rozmiarze ×1,348. Zmierzone, nie oszacowane — ale bezużyteczne, patrz 3.8.

**[Z-]** Zastrzeżenie: pomiary biegły przy ciepłym cache stron (bezpośrednio po pełnym skanie
głębi), a zbiór `main` ma 15 GB przy ~20 GB wolnego RAM, więc część odczytów mogła iść z pamięci.
Nie ma to wpływu na wniosek, bo margines nad GPU wynosi dwa rzędy wielkości.

### 3.8 Wąskim gardłem jest GPU, a konkretnie `attentionNet` — 96,4 % iteracji **[Z]**

| | ms/iter | próbek/s | 40 000 kroków |
|---|---|---|---|
| model bez AMP | 1 522,97 | 21,0 | 16,92 h |
| model z AMP | 1 513,76 | 21,1 | 16,82 h |
| dataloader (8 workerów) | 12,1 | 2 645 | — |

GPU jest **126× wolniejsze niż dostarczanie danych**. Przepisanie zbioru bez kompresji nie zmieni
nic w czasie treningu — mierzone przyspieszenie 1,67× dotyczy ścieżki, która i tak czeka.

AMP nie daje żadnego zysku (1 523 → 1 514 ms), co było sygnałem, że coś jest nie tak.
Profilowanie per sieć, batch 32, forward+backward:

| sieć | parametry | ms | udział |
|---|---|---|---|
| `rgbdepth` | 16,66 M | 13,3 | 0,9 % |
| `audiodepth` | 8,98 M | 4,4 | 0,3 % |
| `material` | 11,69 M | 6,9 | 0,5 % |
| **`attention`** | **279,58 M** | **1 468,2** | **96,4 %** |

Źródłem są dwie warstwy `nn.Bilinear(512, 512, 512)` w `attentionNet` — po 134,2 M parametrów
każda, razem 279,6 M z 317 M całego modelu. Sama jedna taka warstwa na wejściu (32·16, 512) to
743,4 ms.

Skalowanie po batchu (AMP): 16 → 787,97 ms; 32 → 1 513,42 ms; 64 → 2 936,05 ms. Zależność jest
liniowa, przepustowość stała ~21 próbek/s niezależnie od batcha. To znaczy, że warstwa jest
**compute-bound, nie latency-bound** — większy batch nie amortyzuje kosztu.

### 3.9 `nn.Bilinear` z PyTorcha jest 38,7× wolniejszy od tożsamego `einsum` **[Z]**

`torch.nn.Bilinear` liczy wynik cecha po cesze (pętla `baddbmm` po wymiarze wyjściowym), co przy
`out_features = 512` daje 512 małych jąder GPU. Ta sama funkcja
`y[n,o] = Σ_ij x1[n,i] · W[o,i,j] · x2[n,j] + b[o]` złożona w dwa `einsum`-y:

| | fwd+bwd, (32·16, 512) |
|---|---|
| `nn.Bilinear` | 743,4 ms |
| `einsum` (ta sama funkcja) | **19,2 ms** |

Efekt na pełnym modelu (batch 32, AMP):

| | ms/iter | próbek/s | 40 000 kroków |
|---|---|---|---|
| `nn.Bilinear` (oryginał) | 1 513,50 | 21,1 | **16,82 h** |
| `BilinearEinsum` | **77,57** | **412,5** | **0,86 h** |

**19,5× end-to-end.** Szczyt pamięci bez zmian (6,28 GB).

Dowód tożsamości (`fast_bilinear.verify_equivalence()`, losowe dane, float32):

- forward: maks. różnica bezwzględna 3,24·10⁻⁵ przy skali wyniku 58,9 → **względnie 5,5·10⁻⁷**
- gradienty: maks. różnica 4,88·10⁻³ przy skali 2 786 → **względnie 1,75·10⁻⁶**

To jest szum reasocjacji float32, nie inna funkcja. Warstwa ma te same parametry (`weight`
o kształcie `(out, in1, in2)`, `bias`), te same nazwy w `state_dict`, więc checkpointy są wymienne
w obie strony.

**Status decyzji: wyłączone domyślnie, dostępne pod `--fast-bilinear`.** Choć tożsamość funkcji jest
dowiedziona, jest to odejście od literalnego kodu referencyjnego i decyzja należy do autora pracy,
nie do narzędzia.

### 3.10 `train.py` Paridy nie uruchomi się na Replice **[Z]**

`train.py:75` woła `builder.build_audiodepth()` **bez argumentu**, czyli z domyślnym
`audio_shape = [2, 257, 121]` — kształtem dla mp3d. Dla Repliki wejście ma 166 ramek, więc
spłaszczona warstwa `conv1x1` wychodzi inna: 8·28·17 = **3 808** kanałów zamiast 8·28·11 = **2 464**.

Potwierdzone uruchomieniem:

```
RuntimeError: Given groups=1, weight of size [512, 2464, 1, 1],
expected input[2, 3808, 1, 1] to have 2464 channels, but got 3808 channels
```

`train_condition.py` przekazuje `audio_shape` jawnie i obchodzi to bez dotykania oryginału.

Drugi, nieszkodliwy błąd w tym samym pliku: `create_optimizer` rozpakowuje krotkę do
`net_visualdepth`, a używa `net_rgbdepth` — działa tylko dzięki temu, że `net_rgbdepth` istnieje
w zasięgu globalnym modułu.

### 3.11 Metryki zgodne z implementacją Paridy do 1,5·10⁻⁶ **[Z]**

`metrics.test_matches_parida()` porównuje nową implementację z kopią 1:1
`util.util.compute_errors`. Maksymalna różnica po wszystkich metrykach: **1,494·10⁻⁶**
(RMSE per próbka: dokładnie 0,0).

Rozróżnienie, które trzeba utrzymać w pracy: `train.py` uśrednia RMSE **po próbkach**
(`np.array(errors).mean(0)`), co nie jest RMSE całego zbioru, tylko średnią z pierwiastków.
Nowa implementacja liczy i raportuje **obie** wielkości (`RMSE` po pikselach i `RMSE_per_sample`),
żeby zestawienie z Paridą było możliwe, a liczba do pracy była ta poprawna.

### 3.12 Maska nieciągłości głębi — czułość na próg **[Z]**

Udział pikseli klasyfikowanych jako krawędziowe, mierzony na próbkach ze zbioru testowego:

| próg | % ważnych pikseli |
|---|---|
| 0,05 m/px | 21,54 % |
| **0,10 m/px** (domyślny) | **10,87 %** |
| 0,20 m/px | 5,86 % |

Gradient liczony wyłącznie między pikselami ważnymi (głębia > 0). Piksel sąsiadujący z dziurą
miałby gradient równy własnej głębi — kilka metrów — i trafiłby do maski bez powodu geometrycznego,
zanieczyszczając właśnie tę metrykę, która ma być najczulsza.

Uzasadnienie istnienia tej metryki: teza pracy dotyczy krawędzi i naroży, a to ~11 % kadru.
Poprawa rzędu kilku procent na 11 % pikseli znika w trzecim miejscu po przecinku globalnego RMSE.

---

## 4. Decyzje projektowe i ich uzasadnienia

### 4.1 h5py nie przeżywa `fork()` — obsłużone przez pilnowanie PID

Otwarty uchwyt HDF5 odziedziczony przez workera DataLoadera daje ciche uszkodzenie danych albo
zawieszenie, **bez komunikatu błędu**. Konstruktor datasetu nie otwiera więc żadnego pliku
(indeks budowany jest przez `with h5py.File(...)`, zamykane od razu), a `_handles()` porównuje
zapisany PID z bieżącym. Po forku słownik uchwytów jest **porzucany, nie zamykany** — zamknięcie
odziedziczonego uchwytu rusza wspólny stan biblioteki HDF5 i jest samo w sobie niebezpieczne. **[W]**

### 4.2 Stała liczba kroków gradientu, nie epok

Warunek `cardinal` ma 5 496 próbek treningowych, `all` ma 49 464 — dokładnie 9× więcej. Przy stałej
liczbie epok ten drugi dostałby 9× więcej kroków optymalizacji i wygrałby z tego powodu, a nie
z powodu gęstości kątowej. Wszystkie warunki dostają **40 000 kroków**.

Konsekwencja do wypunktowania w pracy: równoważnik epok waha się od **232,9** (`cardinal`) do
**25,9** (`all`). Różna liczba powtórzeń tej samej próbki to różne ryzyko przeuczenia — dlatego
checkpoint wybierany jest po najlepszym RMSE walidacyjnym, a nie po ostatnim kroku.

### 4.3 Warunek D (`random_4`) jest niezbędny

Bez niego porównanie A (4 kąty) z B (36 kątów) myli dwie rzeczy: gęstość kątową i rozmiar zbioru.
D ma **dokładnie tyle próbek co A** (5 496), ale kąty są losowane per lokalizacja, więc model widzi
w sumie wszystkie 36 orientacji — tylko nie wszystkie z każdego punktu.

- różnica **D − A** izoluje samą różnorodność kątową przy stałej liczbie próbek
- różnica **B − D** izoluje samą ilość danych

### 4.4 Podzbiory kątów pochodzą z tych samych renderów

36 dzieli się bez reszty przez 2, 3, 4, 6, 9, 12 i 18, więc cała krzywa nasycenia 4/6/9/12/18/36
powstaje z podzbiorów już wygenerowanych danych. Żaden punkt krzywej nie wymaga dogenerowania
czegokolwiek na GPU, a wszystkie punkty pochodzą z **dokładnie tych samych renderów** — to wyklucza
tłumaczenie różnicy między punktami szumem generatora.

### 4.5 ECHO2DEPTH jako osobny warunek

W pełnym modelu obraz RGB niesie większość informacji o głębi, a echo dokłada niewiele — efekt
gęstości kątowej może zginąć pod priorem wizualnym. Sama gałąź audio (`net_audiodepth`, bez fuzji
i bez materiału) jest najczystszym testem hipotezy. Implementacja `EchoOnlyModel` to **dokładnie**
`SimpleAudioDepthNet` Paridy wyjęte z fuzji, z interfejsem identycznym jak `AudioVisualModel`, żeby
pętla treningowa i ewaluacja były wspólne.

Koszt: **0,13 h na przebieg** wobec 0,86 h (z `--fast-bilinear`) lub 16,82 h (bez). Przy tym
warunku wąskim gardłem staje się dataloader (4,4 ms GPU wobec 12,1 ms I/O) — jedyny przypadek
w całej macierzy, gdzie wariant bez kompresji miałby sens, ale przy przebiegu trwającym 8 minut
jest to bez znaczenia.

### 4.6 Bez augmentacji pomijamy PIL — bit-identycznie

`transforms.ToTensor()` na obrazie uint8 to dokładnie dzielenie przez 255 z permutacją osi, więc
w trybie val/test (bez `ImageEnhance`) ścieżka torchowa jest **bit-identyczna**, nie przybliżona.
W trybie treningowym używana jest ścieżka PIL wierna `process_image()` Paridy — te same trzy
operacje, w tej samej kolejności, w tych samych zakresach.

### 4.7 Jeden przebieg na wywołanie procesu

Tak samo jak `generate_echo_dataset.py` miał jedną scenę na proces. Powody: przebieg trwa godziny
i musi dać się wznowić; każdy warunek ma izolowany stan CUDA; kolejnością steruje człowiek.
`exp_ctl.py` wykrywa działające przebiegi skanem `/proc` po wierszu polecenia (nie plikiem PID,
który po twardym zabiciu kłamie) i startuje je z `start_new_session=True`, żeby zerwane SSH
ich nie zabiło. SIGTERM jest przechwytywany i zapisuje checkpoint.

---

## 5. Macierz eksperymentów (zdefiniowana, nie uruchomiona)

| id | subset | kątów | próbek train | ep. równ. | izoluje |
|---|---|---|---|---|---|
| A | `cardinal` | 4 | 5 496 | 232,9 | baseline VisualEchoes (Gao 2020) |
| B | `all` | 36 | 49 464 | 25,9 | efekt łączny gęstości i ilości danych |
| **D** | `random_4` | 4 | 5 496 | 232,9 | **sama gęstość** przy liczności równej A |
| C6 | `every_6` | 6 | 8 244 | 155,3 | krzywa nasycenia |
| C9 | `every_4` | 9 | 12 366 | 103,5 | krzywa nasycenia |
| C12 | `every_3` | 12 | 16 488 | 77,6 | krzywa nasycenia |
| C18 | `every_2` | 18 | 24 732 | 51,8 | krzywa nasycenia |
| EA/EB/ED | jw., **echo2depth** | 4/36/4 | jw. | jw. | hipoteza bez priora wizualnego |
| PA/PB | jw., geometria `patched` | 4/36 | jw. | jw. | wpływ domknięcia dziur |

**3 ziarna na warunek** — pojedynczy przebieg nie pozwala orzec o różnicy 2–3 % w RMSE, bo to mieści
się w rozrzucie samej inicjalizacji wag.

Czas całej macierzy (36 przebiegów), według zmierzonych `s/krok`:

| | godzin | dni |
|---|---|---|
| z `--fast-bilinear` | **24,5** | 1,02 |
| bez | **~590** | ~24,6 |

Sama grupa `glowne` + `echo` (18 przebiegów) to odpowiednio ~9 h i ~143 h.

Metryki raportowane per przebieg: standardowe (RMSE, ABS_REL, log10, MAE, δ<1,25ⁿ), **stratyfikowane**
(osobno na pikselach krawędziowych i gładkich) oraz **rozbicie per scena held-out**.

---

## 6. Walidacja szkieletu

- 60 kroków `ED` (echo2depth) i 40 kroków `A` z `--fast-bilinear`, `num_workers=8` — pełna pętla:
  dataloader z workerami, walidacja, metryki stratyfikowane, per scena, checkpoint. Obie ścieżki
  przeszły; katalogi przebiegów usunięte po sprawdzeniu. **[Z]**
- Podgląd 6 próbek PNG (spektrogram binauralny + RGB + głębia) — echo ma wyraźną ścieżkę
  bezpośrednią i odbicia, kanały L/R różnią się, głębia zgodna z RGB. **[Z]**
- Kopie benchmarkowe bez kompresji usunięte.

---

## 7. Czego NIE sprawdzono **[X]**

- **Żaden model nie był trenowany do końca.** Wszystkie liczby o czasie to ekstrapolacja
  ze zmierzonego `s/krok` × 40 000, bez uwzględnienia narzutu walidacji co 1 000 kroków.
- **Nie sprawdzono zbieżności ani jakości wyników** — dwa przebiegi dymne po 40–60 kroków mówią
  wyłącznie o tym, że kod działa, nic o uczeniu.
- **`--fast-bilinear` nie był użyty w pełnym przebiegu.** Tożsamość funkcji jest dowiedziona na
  losowych danych i na 40 krokach; nie ma dowodu, że po 40 000 krokach trajektoria uczenia jest
  statystycznie nierozróżnialna od oryginału. Gdyby wynik miał trafić do pracy, wart rozważenia
  jest jeden przebieg kontrolny A/seed 0 w obu wersjach.
- **Nie zmierzono narzutu walidacji** ani zużycia dysku przez 36 katalogów przebiegów
  (checkpointy pełnego modelu to ~1,3 GB na przebieg przy 317 M parametrów + stan Adama).
- **`opt.enable_cropping`** pozostaje wyłączone i nie było testowane — obrazy mają dokładnie
  128×128, więc kadrowanie z `image_resolution` byłoby operacją pustą albo błędem.

---

## 8. Stan i co blokuje

**Nic nie blokuje startu treningu.** Dwie rzeczy wymagają decyzji autora:

1. **`--fast-bilinear`: 24,5 h wobec ~590 h dla całej macierzy.** Domyślnie wyłączone. Jeśli
   decyzja to pozostanie przy literalnym `nn.Bilinear`, sensowna kolejność to najpierw grupa
   `echo` (9 przebiegów, ~1,3 h łącznie), bo jest tania i jednocześnie najczystsza dowodowo.
2. **`beyond-image-to-depth/` nie jest śledzone przez git** (~600 KB kodu). Do rozważenia, czy
   wendorować tak jak `habitat-lab/` i `sound-spaces/`, razem z wpisem w `THIRD_PARTY_LICENSES.md`.

Podział lokalizacji leży w gitignorowanym `outputs/`, ale odtwarza się deterministycznie z tym samym
odciskiem (`e0bf7547668d9e0a`), więc nie jest to ryzyko dla powtarzalności.

---

## 9. Gdzie leżą dowody

| co | gdzie | w gicie? |
|---|---|---|
| raport weryfikacji, oba warianty | `outputs/ml/verify_loader/{main,patched}/verify_loader.json` | **tak** |
| surowe wyniki benchmarku | `outputs/ml/bench/bench_main.json` | **tak** |
| podział lokalizacji + odcisk | `outputs/ml/splits/replica_locations.json` | **tak** |
| konfiguracja macierzy | `outputs/ml/experiments.json` | **tak** |
| próbki PNG | `outputs/ml/verify_loader/*/samples/` | nie (2,4 MB) |
| przebiegi treningowe | `outputs/ml/runs/` | nie (~1,3 GB/przebieg) |
| test tożsamości bilinear | `ml/fast_bilinear.py::verify_equivalence()` | tak (kod) |
| test zgodności metryk | `ml/metrics.py::test_matches_parida()` | tak (kod) |

`outputs/**` jest domyślnie ignorowane; cztery pliki JSON, na których opierają się liczby w tym
dokumencie, są jawnie odbiałolistowane w `.gitignore` (łącznie ~49 KB) — zgodnie z konwencją
przyjętą wcześniej dla `outputs/measurements/`. PNG-i i przebiegi zostają poza gitem jako duże
i odtwarzalne: `echo_data.py --verify-loader` odtwarza jedne, `exp_ctl.py start` drugie.
