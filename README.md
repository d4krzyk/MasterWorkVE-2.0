# Visual Echoes 2.0 — gęstość kątowa echa w audiowizualnej predykcji głębi

Praca magisterska. Bada, **ile orientacji agenta trzeba wyrenderować akustycznie**, żeby model
przewidujący głębię z obrazu i echa działał dobrze — i czy istnieje punkt, powyżej którego
zagęszczanie siatki przestaje się opłacać.

Praca rozszerza *VisualEchoes* (Gao i in., ECCV 2020), które renderowało echo z **4 kierunków
kardynalnych**, do **36 orientacji co 10°**, i mierzy, co z tego wynika. Warstwa uczenia opiera
się na *Beyond Image to Depth* (Parida i in., CVPR 2021).

## Wynik w jednym akapicie

Gęstość kątowa poprawia predykcję głębi wyraźnie i istotnie, ale **krzywa nasyca się przy 9–12
orientacjach**: przejście z 4 do 9 daje 6,7× więcej niż całe pozostałe rozszerzenie do 36. Przy
stałym koszcie generowania danych opłaca się więc **losować kąty z siatki 9–12 orientacji**
zamiast przybijać je do 4 kierunków. Efekt utrzymuje się w pełnym modelu audiowizualnym
(p = 0,0096) i nie zależy od wariantu geometrii sceny. Osobno badane zadanie pretekstowe
przewidywania orientacji jest rozwiązywalne (MAAE 25,65° wobec 90° losowego), ale **nie przenosi
się** na predykcję głębi — również przy ograniczeniu zbioru docelowego do 10 %.

Komplet obowiązujących liczb: **[`my-operations/docs/STAN_WYNIKOW.md`](my-operations/docs/STAN_WYNIKOW.md)**.
To jest jedyny dokument, z którego należy przepisywać wartości — raporty sesji są dziennikiem
chronologicznym i zawierają liczby później zastąpione.

| dokument | zawiera |
|---|---|
| `docs/STAN_WYNIKOW.md` | **stan obowiązujący** — wszystkie wyniki, bez historii |
| `docs/LICZBY_DO_PRACY.md` | 134 liczby ze statusem, plikiem dowodowym i sekcją kontroli spójności (generowane) |
| `docs/GENERATOR_PARAMS.md` | zamrożona specyfikacja generatora danych — źródło prawdy dla każdego parametru |
| `docs/RYSUNKI.md` | podpisy rysunków i uzasadnienia decyzji wizualnych |
| `docs/MODELE.md` | nazewnictwo modeli |
| `docs/RAPORT_SESJI_*.md` | dziennik chronologiczny z korektami — **nie cytować bez sprawdzenia w `STAN_WYNIKOW.md`** |

---

## 1. Co jest w tym repozytorium

```
my-operations/          ← KOD AUTORA (to jest właściwy wkład)
  ml/                     pakiet fazy uczenia — JEDYNY katalog importowalny jak biblioteka
    dataset/                dataloader HDF5, podziały, filtry kątów
    depth_model/            trening i ewaluacja modelu głębi
    pretext_model/          zadanie pretekstowe orientacji + transfer
    matrix/                 definicja macierzy eksperymentów, kolejki
    analysis/               liczby do pracy, rysunki, galeria
  echo_core/              wspólny rdzeń generowania: symulator, spektrogramy, ścieżki
  generate_echo_dataset.py  generator zbioru (jedna scena na proces)
  echo_ctl.py               pulpit sterujący generowaniem
  diagnose_rlr_noise.py     charakteryzacja szumu silnika akustycznego
  measurements/           pomiary fizyczne (RT60, łatanie geometrii, sweepy)
  diagnostics/            eksperymenty diagnostyczne silnika
  tools/                  narzędzia pomocnicze (m.in. generator materiałów Repliki)
  docs/                   dokumentacja i raporty
  metadata/               prekomputowane grafy nawigacji (graph.pkl, points.txt)
  notebooks/              notatniki eksploracyjne
  replica_material_config.json   materiały akustyczne dla kategorii Repliki

habitat-sim/            ← WENDOROWANE, ZMODYFIKOWANE (patrz §5)
habitat-lab/            ← wendorowane, bez zmian
sound-spaces/           ← wendorowane, bez zmian
beyond-image-to-depth/  ← wendorowane, NIETKNIĘTE (patrz §5.4)

configs/                snapshoty środowiska conda
smoke-tests/            ręczne skrypty sprawdzające, czy symulator w ogóle startuje
outputs/                WSZYSTKO generowane (gitignore, poza plikami dowodowymi)
```

**Zasada podziału:** `my-operations/` to kod autora, reszta to zawendorowane frameworki
badawcze z zachowanymi licencjami (`THIRD_PARTY_LICENSES.md`).

---

## 2. Odtworzenie od zera — kolejność ma znaczenie

Pięć etapów. Etapy 1–2 to instalacja i dane wejściowe (najbardziej kłopotliwe), 3 to generowanie
zbioru (~30 h GPU), 4 to trening (~30 h GPU), 5 to analiza (minuty, bez GPU).

**Jeśli chcesz tylko sprawdzić liczby i rysunki — przejdź od razu do etapu 5.** Pliki dowodowe
(`outputs/ml/**/*.json`) są w repozytorium; nie wymagają ani danych, ani GPU.

### Etap 0 — środowisko

```bash
conda env create -f configs/environment-final.yml   # tworzy env `habitat`
conda activate habitat
```

Python 3.10.14, PyTorch 2.13.0+cu130. Alternatywne snapshoty tego samego środowiska:
`configs/requirements.txt`, `configs/requirements-2026-07-18.txt`, `configs/spec-file.txt` —
sprawdź, który odpowiada twojej instalacji, zanim założysz wersje pakietów.

### Etap 1 — zbudowanie habitat-sim

To jest część, która najczęściej nie działa od pierwszego razu. **Szczegóły w §5.1.**

```bash
cd habitat-sim
python setup.py build_ext --inplace --headless --audio
export PYTHONPATH=$(pwd)/src_python:$PYTHONPATH
```

> **Uwaga o CUDA — przeczytaj przed budowaniem.** Powyższa komenda **nie** ustawia
> `CMAKE_CUDA_ARCHITECTURES` i przełącza `TARGET_HEADLESS` na `ON`. Na sprzęcie użytym w tej pracy
> (RTX 5070 Ti, Blackwell `sm_120`, toolkit CUDA 12.4) trzeba było skonfigurować `cmake`
> bezpośrednio, z `TARGET_HEADLESS=OFF`, `BUILD_WITH_BULLET=ON`, `BUILD_WITH_CUDA=ON`,
> `BUILD_WITH_AUDIO=ON` i **wymuszonym `CMAKE_CUDA_ARCHITECTURES=90`**. Powód: `nvcc` 12.4 nie zna
> `sm_120`, ale architektura 90 (Hopper) generuje PTX, który sterownik kompiluje JIT dla Blackwella
> przy ładowaniu. Przy zmianach w C++ używaj przebudowy przyrostowej, żeby nie stracić działającej
> konfiguracji cmake:
> ```bash
> cmake --build habitat-sim/build --target habitat_sim_bindings -j$(nproc)
> cp habitat-sim/build/RelWithDebInfo/lib/habitat_sim_bindings.cpython-*.so \
>    habitat-sim/src_python/habitat_sim/_ext/
> ```

habitat-lab i sound-spaces instaluje się edytowalnie (`pip install -e .` z każdego katalogu).

Sprawdzenie, czy symulator startuje (uruchamiać **z katalogu głównego repo** — skrypty rozwiązują
ścieżki względem katalogu roboczego):

```bash
python smoke-tests/test_habitat.py
python smoke-tests/test_replica.py
```

### Etap 2 — dane scen

Sceny **nie są** w repozytorium (gitignore). Pobierz zgodnie z `sound-spaces/INSTALLATION.md`
i `habitat-sim/DATASETS.md`, i umieść pod:

```
sound-spaces/data/scene_datasets/replica/<scena>/habitat/
    <scena>_stage.stage_config.json
    mesh_semantic.navmesh
```

Praca używa **18 scen Replica**. Pozycje agenta pochodzą z `my-operations/metadata/replica/<scena>/`:

- **`points.txt` jest źródłem kanonicznym** (tab-separated `id, a, b, c`, gdzie `x = a`, `z = −b`);
- `graph.pkl` to przefiltrowany podzbiór, pre-oczyszczony pod kątem spójności.

> **Wysokość agenta `y` bierze się z `graph.pkl`, nigdy z `pathfinder.snap_point()`.** `snap_point`
> zwraca **powierzchnię navmesha**, która leży ~0,21 m nad podłogą (mediana po 1 740 lokalizacjach,
> maks. 0,49 m), bo navmesh Repliki nie ma zapisanych `NavMeshSettings` i recast odbudowuje go
> z domyślną kwantyzacją. Zmierzone na `office_1`: `y` z grafu odtwarza referencyjne RGB
> z RMSE 0,0125 i 99,98 % pikseli bit-identycznych; `snap_point` daje RMSE 50,05 i 36 %.
> Funkcja `diagnose_rlr_noise.py::load_point_position()` nadal używa `snap_point` — **celowo**,
> żeby zachować odtwarzalność historycznej charakteryzacji. Nie kopiować jej do kodu produkcyjnego.

### Etap 3 — wygenerowanie zbioru ech

Specyfikacja jest **zamrożona** w `my-operations/docs/GENERATOR_PARAMS.md` — to źródło prawdy dla
każdego parametru (500 promieni pośrednich, `threadCount=1`, adaptywne *N* renderów na lokalizację,
materiały akustyczne włączone, wysokość źródła 1,25 m).

```bash
python my-operations/echo_ctl.py                    # pulpit interaktywny
python my-operations/echo_ctl.py status             # stan wszystkich scen (bez GPU)
python my-operations/echo_ctl.py next               # policz następną nieukończoną scenę
python my-operations/echo_ctl.py verify             # kontrola kompletności HDF5
python my-operations/echo_ctl.py --variant patched status   # wariant geometrii
```

`echo_ctl.py` uruchamia sceny **odczepione od terminala** (`start_new_session=True`), więc zerwane
SSH ich nie zabije, i wykrywa już działające generowanie skanem `/proc`, a nie plikiem PID —
ten po twardym zabiciu kłamie. Pojedynczą scenę można policzyć bezpośrednio:

```bash
python my-operations/generate_echo_dataset.py --scene office_1 --resume
python my-operations/generate_echo_dataset.py --status     # nie dotyka GPU
```

Wynik: `outputs/echoes_36deg/<scena>/<scena>.h5` wraz z `decisions.jsonl` i `generate.log`
(~15 GB łącznie) oraz `outputs/echoes_36deg_patched/` (~11 GB, wariant z załatanymi dziurami
w geometrii). Jedna scena na proces systemowy, jeden długo żyjący `Simulator` — patrz §6,
pułapka o wieszaniu GPU.

Kontrola dataloadera przed treningiem:

```bash
python my-operations/ml/dataset/echo_data.py --verify-loader --geometry main
```

### Etap 4 — macierz eksperymentów

**Podział lokalizacji jest zamrożony** — odcisk `e0bf7547668d9e0a`. Nie regenerować: wszystkie
wyniki są z nim porównywalne, a przebudowa unieważnia je bez śladu.

```bash
python my-operations/ml/matrix/ml_ctl.py plan     # plan, godziny, GB, bez uruchamiania
python my-operations/ml/matrix/ml_ctl.py run      # kolejka
python my-operations/ml/matrix/ml_ctl.py status   # stan
```

Zasady nadrzędne, które trzeba znać przed zmianą czegokolwiek:

- **Stała liczba kroków gradientu (40 000), nie epok**, batch 32, we wszystkich warunkach. Warunek
  `cardinal` ma 5 496 próbek, `all` ma 49 464 — przy stałej liczbie epok ten drugi dostałby 9×
  więcej kroków i wygrałby z tego powodu, a wniosek o gęstości byłby nieważny.
- **Checkpoint wybierany po najlepszym RMSE walidacyjnym** na pełnych 36 kątach, nie po ostatnim
  kroku — bo warunki obchodzą swoje zbiory różną liczbę razy (233× vs 26×) i mają różne ryzyko
  przeuczenia.
- `max_depth = 14,104` m (stała Replica z `base_options.py` Paridy).
- Definicja warunków, historia decyzji o liczbie ziaren i grupy świadomie skreślone:
  `my-operations/ml/matrix/experiments.py`.

Kolejka jest odporna na przerwanie: rozpoznaje ukończone kroki po artefaktach (`status.json`),
a nie po własnej księgowości, więc `run` po awarii wznawia od miejsca zatrzymania. Zatrzymuje się
sama, gdy wolne miejsce spadnie poniżej 15 GB.

### Etap 5 — analiza, liczby, rysunki (bez GPU)

```bash
python my-operations/ml/pretext_model/summarize.py   # tabele Modelu 2
python my-operations/ml/analysis/final_results.py    # plik dowodowy sesji
python my-operations/ml/analysis/thesis_numbers.py   # -> docs/LICZBY_DO_PRACY.md
python my-operations/ml/analysis/figures.py          # -> outputs/ml/figures/
python my-operations/ml/analysis/depth_gallery.py    # galeria (wymaga GPU i checkpointów)
```

**Kolejność ma znaczenie**: `final_results.py` produkuje plik, z którego czyta `thesis_numbers.py`.

`thesis_numbers.py` na koniec sprawdza, czy dwie pozycje nie noszą tej samej nazwy z różnych
plików dowodowych, i wypisuje kolizje. Ta kontrola powstała po realnym błędzie — patrz §7.

---

## 3. Struktura wyników

| katalog | zawartość |
|---|---|
| `outputs/echoes_36deg{,_patched}/` | zbiór ech, HDF5 na scenę (gitignore, ~26 GB) |
| `outputs/ml/runs/<warunek>_seed<n>/` | wagi i logi treningu (gitignore) |
| `outputs/ml/eval/<przebieg>/` | metryki testowe, `eval.json` **jest** w repo |
| `outputs/ml/echo_ablation/*.json` | pliki dowodowe wyników — **w repo** |
| `outputs/ml/figures/*.png` | rysunki do pracy — **w repo** |

Wersjonowane są **liczby i rysunki**, nie wagi. `.gitignore` wymienia wyjątki jawnie, z powodem
przy każdym.

---

## 4. Sprzęt, na którym to powstało

RTX 5070 Ti (16 GB, Blackwell), sterownik 580.159.03+, toolkit CUDA 12.4. Czasy zmierzone:

| operacja | czas |
|---|---|
| render akustyczny (materiały włączone) | 0,111–0,148 s |
| przebieg treningowy, sieć tylko-echo | ~10,5 min |
| przebieg treningowy, model pełny | ~57 min |
| przebieg zadania pretekstowego | ~12 min |
| cała macierz, orientacyjnie | ~30 h GPU |

---

## 5. Co zrobiono z zawendorowanymi bibliotekami

### 5.1 habitat-sim — **zmodyfikowany**, cztery zmiany

Przypięty do commita `80f8e31140eaf50fe6c5ab488525ae1bdf250bd9`
(`habitat-sim/COMMIT_HASH.txt`). Ma **własny katalog `.git`**, więc `git` z katalogu głównego
operuje na projekcie zewnętrznym — żeby zobaczyć jego historię, użyj `git -C habitat-sim`.
Wszystkie modyfikacje są zebrane w `habitat-sim/local_changes.patch`.

**(1) Kolejność importów.** `quaternion` jest importowany **przed** rozszerzeniem
`habitat_sim_bindings` w `src_python/habitat_sim/__init__.py`. Bez tego proces kończy się
`free(): invalid pointer` (facebookresearch/habitat-sim#1747). Jeśli piszesz kod, który omija ten
`__init__.py`, musisz zrobić to samo ręcznie: `import quaternion` przed `import habitat_sim`.

**(2) `AudioSensor::loadMesh()` faktycznie wysyła siatkę.** Oryginał **nigdy nie wołał**
`audioSimulator_->UploadMesh()`. Dodane, wraz z logowaniem kodów błędu zwracanych przez
`Configure`/`AddListener`/`LoadMeshData`/`UploadMesh`/`AddSource` — bez tego awarie silnika były
niewidoczne.

**(3) Podbicie `rlr-audio-propagation` z `bdb262d` na `4fd446b`.** To jest zagnieżdżony submoduł
(ma własny `.git`). Pierwotnie przypięta wersja z lipca 2022 ma realny błąd: przy
`enableMaterials=False` `UploadMesh()` **deterministycznie** zwraca
`ErrorCodes::MemoryAllocFailure` (2018) niezależnie od rozmiaru siatki, konfiguracji akustyki
i zasobów systemu (potwierdzone `strace` — zero `ENOMEM`, pamięci pod dostatkiem), przez co
`getIR()` zawsze zwracał pustkę. Nowsza wersja (`main`, ten sam cel co PR #1923 w habitat-sim)
naprawia to. Nowe API oznacza starą klasę `Simulator`/`Configuration`/`ChannelLayout` jako
`RLRA_DEPRECATED`, więc każde jej użycie w `AudioSensor.{h,cpp}` i w bindingach pybind11
(`src/esp/bindings/SensorBindings.cpp`) jest otoczone lokalnym
`#pragma GCC diagnostic push/ignored("-Wdeprecated-declarations")/pop`. Build jest
**bezostrzeżeniowy** (`-Wall -Werror`, 0 błędów, 0 ostrzeżeń).

**(4) Naprawa SIGSEGV w `AudioSensor::loadSemanticMesh()`.** Włączenie materiałów akustycznych
wywalało symulator na **każdej** scenie Replica. Przyczyna: `info_semantic.json` Repliki deklaruje
obiekty z `class_id: -1` (12 z 92 w samym `room_0`), a kontrola
`categoryIndex < categories_.size()` w `ReplicaSemanticScene.cpp` porównuje `int` ze znakiem
z `size_t` bez znaku — `-1` przepełnia się i porównanie przypadkiem pomija przypisanie kategorii.
Powstaje **niepusty `SemanticObject` z pustym `category()`**. Stary kod sprawdzał tylko obiekt,
po czym bezwarunkowo łuskał `->category()->name()`. Dodano sprawdzenie przez pomocnicze
`categoryNameForVertex`. Wcześniejsza hipoteza o indeksach poza zakresem była **błędna** —
instrumentowana przebudowa potwierdziła, że wszystkie zakresy obiektów, wierzchołków i indeksów
są poprawne.

> **Materiały akustyczne wymagają dwóch rzeczy naraz:** `acousticsConfig.enableMaterials = True`
> **oraz** `cfg.load_semantic_mesh = True`. `setAudioMaterialsJSON()` działa wyłącznie na ścieżce
> `loadSemanticMesh()`. Wzorzec działający: `my-operations/echo_core/audio.py::build_simulator()`;
> najkrótsze sprawdzenie od końca do końca: `my-operations/smoke_test_rlr_audio.py`.
>
> Materiały dla Repliki są w `my-operations/replica_material_config.json`, generowanym przez
> `my-operations/tools/make_replica_material_config.py` — **nie** używaj
> `sound-spaces/data/mp3d_material_config.json`, bo jest zbudowany na kategoriach Matterport3D
> i na scenach Replica wypisuje setki linii „Material for category 'X' was not found. Using default
> material instead." Fallback jest nieszkodliwy, ale wtedy praktycznie cała scena dostaje materiał
> domyślny, czyli materiały są formalnie włączone, a akustycznie nieobecne.

### 5.2 habitat-lab — bez zmian

Wersja z linii v0.2.2. Zawendorowany jako **zwykły katalog**, bez zagnieżdżonego `.git`, więc jego
historia nie jest osobno śledzona. Traktuj wszelkie różnice wobec upstreamu jako celowe zmiany
autora.

### 5.3 sound-spaces — bez zmian

SoundSpaces 1.0 + 2.0, również zwykły katalog. Używany do konwencji układu danych scen i jako
odniesienie instalacyjne. Poprawność sprawdza się ręcznie:
`python examples/minimal_example.py` powinno wyprodukować `data/output.wav`.

### 5.4 beyond-image-to-depth — **NIETKNIĘTY, celowo**

Przypięty do `dcdef5122fa456a92bd58ead4eea0a777158c535`. Ma własny `.git`, ale repozytorium
zewnętrzne śledzi jego 25 plików **bezpośrednio**, nie jako submoduł.

> **Jego plików nie wolno zmieniać.** Zmienną niezależną tego badania jest gęstość kątowa echa.
> Zmiana sieci, straty czy optymalizatora w implementacji referencyjnej sprawiłaby, że efektu nie
> dałoby się przypisać. Nowy kod trafia do `my-operations/ml/` i **importuje** sieci Paridy.
> Kontrola: `git diff beyond-image-to-depth/` musi być pusty.

W kodzie upstreamu są **dwa realne błędy**. Oba są **obchodzone** w
`my-operations/ml/depth_model/train_condition.py`, a nie łatane u źródła:

1. `train.py` woła `builder.build_audiodepth()` bez argumentu, czyli z domyślnym dla mp3d
   `audio_shape=[2, 257, 121]`. Na 166-ramkowym wejściu Repliki wywala się: `conv1x1` oczekuje
   2 464 kanałów, dostaje 3 808.
2. `create_optimizer` rozpakowuje `net_visualdepth`, ale używa modułowo-globalnego `net_rgbdepth`.

> **Uwaga przy dodawaniu plików z katalogu mającego zagnieżdżony `.git`:** git po cichu utworzy
> *gitlink* zamiast dodać pliki. Obejście przez tymczasową zmianę nazwy `.git` działa, ale wtedy
> przemianowany katalog przestaje być chroniony — dlatego `.gitignore` pilnuje `**/.git.disabled/`.

---

## 6. Pułapki, które kosztowały czas

- **Nie twórz wielu `Simulator`ów w jednym procesie.** Po ~30 konstrukcjach/destrukcjach
  obserwowano natywną awarię renderera (asercja kompletności framebuffera w Magnum/GL) — wyciek
  zasobów EGL/GL przy `sim.close()`. To potrafi **zawiesić GPU sprzętowo**: `nvidia-smi` raportuje
  wtedy `pstate: [GPU requires reset]`, a tworzenie kontekstu CUDA/EGL zawodzi
  (`cuInit` → `CUDA_ERROR_NO_DEVICE`). Ani `nvidia-smi -r`, ani przeładowanie modułów, ani PCI
  `remove`+`rescan` tego nie czyszczą. Działa dopiero **prawdziwy Function-Level Reset**:
  `echo 1 | sudo tee /sys/bus/pci/devices/0000:01:00.0/reset` (po wyładowaniu modułów `nvidia*`).
  **Dlatego generator dzieli pracę na osobne procesy systemowe** — jedna scena na proces.
- **Niezgodność wersji sterownika bez restartu.** Po aktualizacji NVIDII przez `apt`, ale bez
  przeładowania modułu jądra, dostaniesz `unable to find EGL device for CUDA device 0` (odtwarza
  się nawet z gołym sensorem RGB, bez sceny i bez audio). Sprawdź `cat /sys/module/nvidia/version`
  wobec `dpkg -l | grep nvidia-driver`. Da się naprawić bez restartu:
  `sudo systemctl isolate multi-user.target`, potem `rmmod` i `modprobe` modułów
  `nvidia_drm nvidia_uvm nvidia_modeset nvidia` w tej kolejności, potem
  `sudo systemctl isolate graphical.target`. Samo zatrzymanie menedżera logowania nie wystarczy —
  `nvidia_drm` obsługuje też framebuffer konsoli i pozostaje zajęty.
- **Renderowanie programowe** (WSL, brak GPU): ustaw `GALLIUM_DRIVER=llvmpipe`
  i `MESA_GL_VERSION_OVERRIDE=4.1` przed utworzeniem `Simulator`. Pod WSL ma też znaczenie
  `cfg.gpu_device_id = 0`.
- **Szum symulacji akustycznej jest duży.** Ścieżka bezpośrednia jest deterministyczna, ale całe
  wahanie między renderami siedzi w stochastycznych odbiciach pośrednich (Monte Carlo). Zmierzone:
  RMSE spektrogramu między renderami odległymi o 10° ≈ 0,06, o 90° ≈ 0,30–0,35, a **czysty szum
  render-do-renderu tego samego kąta to 0,03–0,16**, zależnie od pozycji. Przy pojedynczym
  renderze szum bywa **3× większy niż sygnał 10°**, który ma mierzyć — stąd adaptacyjna liczba
  renderów uśrednianych na lokalizację (patrz `GENERATOR_PARAMS.md` §3).

## 7. Uwagi metodologiczne warte przeczytania przed modyfikacją

Trzy rzeczy, które w tym projekcie okazały się ważniejsze, niż wyglądały:

**Agregacja musi grupować po warunku, nie po jego części.** Realny błąd: tabela grupowała
przebiegi zadania pretekstowego po `K`, przez co warunek `K=36` i jego **własna kontrola**
`K=36 @ 16 par` wpadały do jednego kubełka. Wychodziło „25,13 i 61,77 → 43,45 ± 25,91", podane
tak, jakby to był rozrzut po ziarnach. Liczba nie istniała. Stąd kontrola
`thesis_numbers._duplikaty_nazw()` i klucze nazwane po **wariancie**, nie po `K`.

**Jedno ziarno nie wystarcza — sprawdzone empirycznie, nie z zasady.** Każde twierdzenie oparte
na jednym ziarnie, które przeliczono później na trzy, wymagało korekty: MAAE dla K=4 i K=12
okazały się nierozróżnialne, znak jednego z kontrastów geometrii się odwrócił, a teza o odporności
wyniku na wybór maski upadła. Żadne nie było całkiem błędne — żadne nie mówiło dokładnie tego, co
mówiło.

**Reguły zapisane przed pomiarem zostają w dokumentacji także wtedy, gdy okażą się złe.**
Przykład jest w `experiments.py::SEED_DECISIONS`: przedrejestrowana heurystyka kazała ograniczyć
grupę do jednego ziarna, bo przewidywany efekt wypadał poniżej progu. Pomiar pokazał, że proxy
**zaniżyło efekt 3,87×**. Decyzję zmieniono, ale pierwotna reguła, jej uzasadnienie i kolejność obu
decyzji są zapisane w kodzie. Zawiodła nie reguła, tylko jej przesłanka — i to jest ta różnica,
którą trzeba móc odtworzyć po fakcie.

---

## 8. Licencje

`THIRD_PARTY_LICENSES.md` dokumentuje warunki zawendorowanych projektów: habitat-sim i habitat-lab
(MIT), sound-spaces (CC-BY-4.0), beyond-image-to-depth (MIT). Oryginalne noty licencyjne
i copyrightowe są zachowane. Modyfikacje autora są objęte osobnym copyrightem.

## Odniesienia

- Gao i in., *VisualEchoes: Spatial Image Representation Learning through Echolocation*, ECCV 2020
- Parida i in., *Beyond Image to Depth: Improving Depth Prediction using Echoes*, CVPR 2021
- Chen i in., *SoundSpaces: Audio-Visual Navigation in 3D Environments*, ECCV 2020
- Straub i in., *The Replica Dataset: A Digital Replica of Indoor Spaces*, 2019
