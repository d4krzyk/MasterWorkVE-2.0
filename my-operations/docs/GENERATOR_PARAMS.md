# Parametry generatora datasetu — FINALNE

Specyfikacja zamrożona **2026-07-26**, po zamknięciu fazy charakteryzacji i walidacji.
To jest dokument referencyjny dla sesji, w której powstanie generator — **nie jest to kod generatora**.

Każda pozycja ma odwołanie do eksperymentu, który ją rozstrzygnął. Surowe wyniki:
`outputs/diagnose_rlr_noise_out/diagnostics_report.json` (klucz podany przy każdej pozycji).
Konteksty: `docs/PKL_FORMAT.md` (kamera i zbiór lokalizacji), `docs/REPLICA_MATERIALS.md` (materiały akustyczne),
`CLAUDE.md` (środowisko, pułapki GPU).

---

## 1. Tabela parametrów

| parametr | wartość | źródło |
|---|---|---|
| `indirect_ray_count` | **500** | `e2_bias_orientation`, `e2_rays_vs_renders` |
| `thread_count` | **1** | `e2_thread_budget_confirm` |
| `n_renders` | **adaptacyjne per lokalizacja**, `N ∈ [6, 40]`, patrz §3 | `noise_floor_scenes`, `noise_floor_orientation` |
| `averaging_domain` | **`"mag"`** = (1/N)Σ\|STFT\| | `e3_averaging_domain` |
| `material_config` | **`my-operations/replica_material_config.json`** | `materials_verify` |
| `listener_height` | **1.25 m** (kamera i audio) | `listener_height`, `PKL_FORMAT.md` |
| `simulator_rotation` | **nie dotyczy** — brak wycieku | `gpu_memory_scale` |
| `spectrogram_dtype` | **`float16`** na dysku, akumulacja w `float32` | §4.1 |
| `depth_dtype` | **`float32`** (celowo nie float16) | §4.1 |
| `est_time_total` | **≈ 45 h** (+ dorenderowanie, §3.4) | §4 |
| `est_disk` | **≈ 19 GB** | §4 |

---

## 2. Konfiguracja symulatora

```
SimulatorConfiguration
  scene_id            = sound-spaces/data/scene_datasets/replica/<scena>/habitat/mesh_semantic.ply
  load_semantic_mesh  = True          # wymagane, inaczej materiały są cicho ignorowane
  enable_physics      = False
  create_renderer     = True          # bez tego PTex krasuje jeszcze przed audio
  gpu_device_id       = 0

CameraSensorSpec  rgb    : COLOR, 128x128, hfov 90, position [0, 1.25, 0]
CameraSensorSpec  depth  : DEPTH, 128x128, hfov 90, position [0, 1.25, 0]
AudioSensorSpec   audio  : position [0, 1.25, 0]
  acousticsConfig.sampleRate        = 44100
  acousticsConfig.enableMaterials   = True
  acousticsConfig.indirectRayCount  = 500
  acousticsConfig.threadCount       = 1
  channelLayout.channelType         = Binaural
  channelLayout.channelCount        = 2

setAudioMaterialsJSON(my-operations/replica_material_config.json)
```

Obserwacje wizualne zapisywać **surowe** z `get_sensor_observations()` — bez normalizacji i bez klipowania
głębi (`PKL_FORMAT.md`: pkl trzyma RGBA uint8 i głębię w surowych metrach).

### Potok spektrogramu (bez zmian, `test_rlr_audio.render_spectrogram`)

```
SAMPLE_RATE = 44100 ;  ECHO_MS = 60  ->  ECHO_SAMPLES = 2646
STFT_N_FFT = 512 ;  STFT_WIN_LENGTH = 64 ;  STFT_HOP_LENGTH = 16
chirp: my-operations/sweep_audio/3ms_sweep.wav
spektrogram: (2, 257, 166)  -- float32 w pamieci, float16 na dysku (patrz 4.1)
```

### Zbiór próbek

- **Lokalizacje: 1740** — klucze `scene_observations_128.pkl`, nie `points.txt` w całości. To zbiór
  odpowiadający próbkowaniu z pracy Gao (VisualEchoes, ECCV 2020), więc tylko on daje porównywalność.
- Współrzędne z `points.txt`: `x = a`, `z = -b`, `y` z `pathfinder.snap_point([x, y_guess, z])`.
  `location_id` w pkl **jest** kolumną `id` z `points.txt`, a współrzędne obu źródeł są identyczne co do bitu
  (`PKL_FORMAT.md`).
- **Orientacje: 36**, co 10°, `quat_from_angle_axis(deg2rad(kąt), [0, 1, 0])`.
- **Razem 62 640 próbek** (1740 × 36).

Rozkład lokalizacji po scenach (największa `apartment_0` = 211, najmniejsza `office_1` = 16) —
patrz `diagnostics_report.json → gpu_memory_scale`.

---

## 3. Adaptacyjne N

### 3.1 Dlaczego nie stałe N

Dwa zmierzone fakty, które razem wykluczają stałą liczbę renderów jako sensowny wybór:

- **sygnał 10° jest niezależny od sceny**: RMSE spektrogramu 0.0639–0.0662 na 8 pozycjach w 4 scenach
  held-out (`noise_floor_scenes`), mediana 0.0644;
- **podłoga szumu Monte Carlo waha się 2.7×**: 0.029–0.090, i jest własnością **pozycji**, nie orientacji —
  rozrzut wewnątrz pozycji 5.5 % wobec 84 % między pozycjami (`noise_floor_orientation`).

Stałe N musi więc albo marnować rendery tam, gdzie jest łatwo, albo nie osiągać założonego SNR tam, gdzie
jest trudno.

Rozkład wymaganego `N_raw` po 12 zmierzonych pozycjach — **4, 6, 7, 7, 9, 9, 9, 10, 11, 12, 13, 21**,
średnia **9.83**. Podstawa: `noise_floor_scenes` (8 pozycji, 1.25 m) i `signal_noise_recheck` z configiem
`replica` (4 pozycje, 1.5 m), po 2 orientacje na pozycję; dla dwóch pozycji, dla których istnieje pełny obrót
(`noise_floor_orientation`, 36 orientacji), wzięto **medianę po orientacjach** zamiast wartości z dwóch kątów
— jest to lepsze oszacowanie tej samej wielkości. Bez tej podmiany lista brzmi 3, 6, 7, 7, 9, 9, 9, 10, 11,
12, 13, 19 (średnia 9.58); różnica nie zmienia żadnego wniosku.

| schemat | czas | pokrycie |
|---|---|---|
| **adaptacyjne** | **44.6 h** | **12/12** |
| stałe N=12 | 54.4 h | 10/12 |
| stałe N=18 | 81.6 h | 11/12 |
| stałe N=21 | 95.2 h | 12/12 |

Adaptacyjne jest **2.14× tańsze niż stałe N o tej samej gwarancji**. Jest też odporne na to, że rozkład znamy
z 12 z 1740 lokalizacji — samo dostosuje się do pozycji, których nie zmierzyliśmy. Stałe N nie jest.

### 3.2 Kryterium

Wszystko liczone z renderów, które i tak powstają — **żaden dodatkowy pomiar wstępny nie jest potrzebny**.
Estymator szumu to ten sam podział na dwie rozłączne połówki, którego używa każdy eksperyment w tym projekcie:

```
sigma_1 = RMSE(polowka_A, polowka_B) / sqrt(2) * sqrt(n/2)     # szum POJEDYNCZEGO renderu
N       = clamp( ceil( (TARGET_SNR * sigma_1 / SIGNAL_10DEG)^2 ), N_MIN, N_MAX )

TARGET_SNR   = 3.5
SIGNAL_10DEG = 0.0644      # mediana z noise_floor_scenes, stała między scenami
N_MIN, N_MAX = 6, 40
```

`RMSE(A, B) = sqrt(2)·sigma_N` — stąd dzielenie przez `sqrt(2)`. Nigdy nie porównywać surowych RMSE dwóch
zaszumionych estymat bez tej dekompozycji (błąd popełniony raz w Bloku B).

#### Dlaczego `N_MAX = 40`, a nie 24 (rewizja 2026-07-26)

Pierwotne `N_MAX = 24` odpowiada progowi `sigma_1 = sqrt(24)·0.0644/3.5 = 0.09014`. Przegląd wszystkich
dotychczasowych pomiarów pokazał, że ten próg jest ustawiony **dokładnie na krawędzi zmierzonych danych**:

- z 12 pozycji o zmierzonym szumie (`noise_floor_scenes` — 8 pozycji przy 1.25 m; `signal_noise_recheck`
  z configiem `replica` — 4 pozycje przy 1.5 m) **żadna** nie przekracza `N_raw = 24`. Rozkład `N_raw`:
  **3, 6, 7, 7, 9, 9, 9, 10, 11, 12, 13, 19** (przed podmianą dwóch pozycji na mediany z pełnego obrotu, patrz §3.1);
- ale na poziomie pojedynczych orientacji (`noise_floor_orientation`, 72 pomiary) najgorsza wypada przy
  `sigma_1 = 0.09006`, czyli **0.09 % poniżej progu obcięcia**. Margines jest zerowy.

Rozstrzygający jest jednak argument spoza tej próbki: `CLAUDE.md` dokumentuje z charakteryzacji z 07-20
zakres szumu render-do-renderu **0.03–0.16** RMSE między dwoma pojedynczymi renderami, czyli
`sigma_1` do `0.16/sqrt(2) = 0.1131` → **`N_raw` do 38**. `N_MAX = 24` obcinałby takie pozycje o ~40 %
wymaganej liczby renderów, i to po cichu.

`N_MAX = 40` pokrywa `sigma_1` do `sqrt(40)·0.0644/3.5 = 0.1164`, czyli **cały udokumentowany zakres**.

Koszt jest znikomy, bo clamp dotyczy wyłącznie ogona rozkładu — nie zmienia N dla żadnej z 12 zmierzonych
pozycji. Dodatkowy czas względem `N_MAX = 24`:

| zakładany udział pozycji z `N_raw > 24` | +czas dla `N_MAX = 40` |
|---|---|
| 0.5 % | 0.18 h |
| 1 % | 0.36 h |
| 2 % | 0.73 h |
| 5 % | 1.81 h |
| 22.1 % (górna granica CI95 przy 0/12) | 8.02 h |

Oszacowanie udziału ogona — **zgrubne, z próbki 12 pozycji na 1740, dobranej wygodnie (ułamki 0.20/0.75
długości `points.txt`), a nie losowo**: dopasowanie lognormalne daje 1.28 % powyżej progu 24 i 0.05 % powyżej
progu 40; dopasowanie normalne odpowiednio 0.18 % i 0.00 %. Przy 0 przekroczeniach na 12 prób dane są jednak
formalnie zgodne z ogonem sięgającym 22 % (górna granica ufności 95 %), więc żadnej z tych liczb nie należy
traktować jako twierdzenia — stąd decyzja oparta na udokumentowanym zakresie szumu, a nie na ekstrapolacji.

**Clamp nie jest narzędziem kontroli budżetu, tylko bezpiecznikiem** przed patologiczną geometrią. Każde jego
zadziałanie musi zostawić ślad — patrz `clamped` w §3.4.1 i ograniczenie 6 w §5.

### 3.3 Procedura per lokalizacja

1. **Sonda**: 8 renderów przy orientacji 0°, podział 4+4 → `sigma_1` → `N`.
2. Renderuj `N` dla **wszystkich 36 orientacji** tej lokalizacji, **wykorzystując rendery sondy** dla
   orientacji 0°. Odzyskiwane w całości gdy `N ≥ 8`; strata gdy `N < 8` to w sumie **0.06 h** na cały zbiór.
   Gdy `N < 8`, dla orientacji 0° użyć **pierwszych `N`** renderów sondy i odrzucić resztę — dzięki temu `N`
   jest jednolite w obrębie lokalizacji. (Nie jest to wymóg poprawności — obciążenie estymatora `mag` nie
   zależy od `N`, patrz E2 — tylko spójności opisu; generator nie powinien tego "naprawiać" inaczej.)
3. Estymata próbki = `mean(|STFT|)` po `N` renderach (domena `mag`), akumulowana w `float32`, zapisywana
   w `float16` (§4.1).

Decyzja zapada **raz na lokalizację** (1740 decyzji, nie 62 640). To jest legalne, bo szum jest własnością
pozycji (§3.1), i dodatkowo prawie eliminuje obciążenie od *optional stopping* — regułę stosuje się na jednej
próbce i przenosi na 36, zamiast zatrzymywać się 62 640 razy w momencie, gdy oszacowanie szumu akurat wypadnie
nisko.

### 3.4 Weryfikacja po fakcie zamiast marginesu bezpieczeństwa

Po zakończeniu każdej próbki policzyć **osiągnięty** SNR z finalnego podziału na połówki — jest darmowy, bo
wszystkie `N` renderów są w pamięci. Zapisać go w metadanych próbki i **dorenderować te, które nie dobiły**
do 3.5.

To jest lepsze niż ślepe zawyżanie `N` „na wszelki wypadek": oszacowanie `sigma_1` z 8 renderów ma ~10 % błędu,
co przekłada się na ~20 % błędu `N`, a margines pokrywający to zjadłby przewagę czasową. Weryfikacja po fakcie
jest dokładna i kosztuje tylko dorenderowanie nielicznych przypadków.

Efekt uboczny wart osobnego zdania w pracy: dostajesz **zmierzony rozkład osiągniętego SNR** per próbka zamiast
deklaracji.

### 3.4.1 Schemat metadanych próbki

Zapisanie tylko finalnego SNR dałoby rozkład **ocenzurowany od dołu**: z definicji nic poniżej 3.5, więc nie
dałoby się odróżnić „reguła trafia dokładnie" od „reguła chybia, ale dorenderowanie to naprawia". To są dwie
różne rzeczy i obie są interesujące — pierwsza mówi o jakości estymatora `sigma_1` z 8-renderowej sondy, druga
jest gwarancją jakości datasetu. Dlatego **dwa niezależne pola**.

Każda próbka `(scena, location_id, orientacja)`:

| pole | typ | znaczenie |
|---|---|---|
| `n_probe` | int | rendery sondy użyte do wyznaczenia `N` (8, chyba że `N < 8` — patrz §3.3 pkt 2) |
| `sigma_1_probe` | float | `sigma_1` oszacowane z sondy; wejście do wzoru na `N` |
| `n_planned` | int | `N` po clamp — liczba renderów zaplanowana z sondy |
| `snr_probe` | float | SNR osiągnięty **po `n_planned` renderach, przed dorenderowaniem** |
| `n_rendered_extra` | int | ile renderów dołożono w kroku weryfikacji (0, jeśli sonda trafiła) |
| `n_total` | int | `n_planned + n_rendered_extra` — faktyczna liczba renderów w estymacie |
| `snr_final` | float | SNR po dorenderowaniu; **musi być ≥ 3.5** |
| `clamped` | str | `""` \| `"min"` \| `"max"` — czy `N_raw` zostało obcięte przez `N_MIN`/`N_MAX` |
| `n_raw` | int | `N` **przed** clampem; pozwala policzyć, jak często i o ile clamp zadziałał |

`snr_probe` i `snr_final` liczone tym samym wzorem, z podziału na dwie rozłączne połówki dostępnych renderów:
`snr = SIGNAL_10DEG / (RMSE(A, B) / sqrt(2))`. Oba są darmowe — rendery i tak są w pamięci.

Reguła spójności, którą warto sprawdzić po generacji: `n_rendered_extra > 0` **wtedy i tylko wtedy**, gdy
`snr_probe < 3.5`. Rozbieżność oznacza błąd w pętli weryfikacji.

Próbki z `clamped == "max"` i `snr_final < 3.5` to jedyne, dla których gwarancja jakości nie obowiązuje — one
muszą trafić do raportu w pracy (§5, ograniczenie 6). Przy `N_MAX = 40` oczekujemy ich zero, ale to ma być
stwierdzone pomiarem, nie założeniem.

### 3.5 Sformułowanie do pracy

> Sygnał różnicujący orientacje jest niezależny od sceny (RMSE spektrogramu 0.0639–0.0662 na ośmiu pozycjach w
> czterech scenach testowych), natomiast podłoga szumu Monte Carlo waha się 2.7-krotnie i jest własnością
> pozycji nasłuchu, a nie orientacji (rozrzut wewnątrz pozycji 5.5 % wobec 84 % między pozycjami). Stała liczba
> renderów musi więc albo marnować obliczenia tam, gdzie jest łatwo, albo nie osiągać założonego SNR tam, gdzie
> jest trudno. Zastosowano zamiast tego adaptacyjne próbkowanie sterowane wariancją — technikę standardową w
> renderingu Monte Carlo — z progiem wyznaczanym per lokalizacja z podziału renderów na dwie niezależne
> połówki, co daje jednorodne SNR ≥ 3.5 przy 2.14-krotnie mniejszym koszcie niż stała liczba renderów o tej
> samej gwarancji.
>
> Zbiór raportuje **dwa** rozkłady SNR: `snr_probe` — osiągnięty po liczbie renderów przewidzianej przez regułę,
> przed jakąkolwiek korektą, będący miarą trafności samego estymatora — oraz `snr_final`, po uzupełnieniu
> nielicznych próbek, które nie dobiły progu, będący gwarancją jakości zbioru. Rozdzielenie tych dwóch
> wielkości jest konieczne, ponieważ sam `snr_final` jest z definicji ocenzurowany od dołu i nie pozwala
> ocenić, jak dokładna była reguła.

---

## 4. Architektura wykonania i budżet

- **Jeden długo żyjący `Simulator` na scenę.** Restart odtwarza identyczną sekwencję RNG, więc **nigdy nie
  restartować per render** (`e1`, `e1_extended`).
- **Świeży proces OS na scenę** (18 procesów). Powyżej ~30 konstrukcji `Simulator` w jednym procesie karta
  zawiesza się sprzętowo — procedura odzysku przez PCI FLR w `CLAUDE.md`.
- **Bez rotacji instancji w obrębie sceny.** `gpu_memory_scale`: 3000 renderów w jednej instancji, RSS
  dokładnie płaskie (1268 MiB, +0.0 MiB/1000), GPU bez trendu (892–984 MiB przy rozrzucie pomiaru 19.5 MiB),
  czas renderu płaski ~0.29 s. Największa scena to `apartment_0` ≈ 76 tys. renderów w jednej instancji.
- **Checkpoint na granicy próbki jest BEZPIECZNY** (`e1_checkpoint_boundary_merge`, R=16: mediana |r| 0.0222
  vs 0.0205, przekroczenia 22/128 vs 22/128, Wilcoxon p=0.949). Wznawianie po awarii nie koreluje szumu.

Tempo: **0.2606 s/render**, średnia ważona liczbą próbek po 18 scenach. Rozmiar sceny prawie nie wpływa —
cały rozrzut to 1.6× (`apartment_0` 0.3531 s, `frl_apartment_5` 0.2205 s).

Budżet:

```
czas  = 62 640 probek x 9.83 renderu (srednia po clamp [6,40]) x 0.2606 s  ~ 44.6 h
        + dorenderowanie z 3.4 + ogon N_MAX (~0.2-0.7 h, patrz 3.2)        ~ 45 h
dysk  = 10.7 GB spektrogramow float16 (2,257,166)
      +  8.2 GB RGB uint8 + depth float32 (128x128)
      = 18.9 GB                                  (wolne na /home: 282 GB)
```

### 4.1 Format zapisu: `float16` dla spektrogramów

Zweryfikowane 2026-07-26 na **uśrednionych estymatach** (`mag`, N=10 na połówkę), czyli na tym, co faktycznie
trafia na dysk — nie na pojedynczych renderach. Sześć próbek z czterech scen o różnej akustyce
(`hotel_0` id=76 i 20, `frl_apartment_5` id=186 i 49, `office_4` id=26, `room_0` id=50):

| | wartość |
|---|---|
| błąd kwantyzacji `float32→float16→float32`, RMSE | **7.2·10⁻⁵ – 8.3·10⁻⁵** |
| największy pojedynczy błąd (na szczycie) | 1.89·10⁻³ |
| szum estymaty `sigma_estimate` tych samych próbek | 0.0085 – 0.0199 |
| **stosunek szum / błąd kwantyzacji** | **112× – 240×** |

Zakres wartości — sprawdzony osobno, bo to on decyduje o obcinaniu, a nie sam błąd RMSE:

- maksimum spektrogramu: **5.1264** we wszystkich sześciu próbkach (to szczyt ścieżki bezpośredniej, ta sama
  w każdej scenie, bo źródło jest współlokowane z odbiornikiem). Zapas do `float16` max 65504 to **12 770×**.
  Zero komórek powyżej zakresu.
- minimum niezerowe: **7.9·10⁻⁸ – 8.3·10⁻⁸**, powyżej najmniejszej denormalnej `float16` (5.96·10⁻⁸).
  **Zero komórek spłaszczonych do zera** we wszystkich sześciu próbkach.
- 686–1009 komórek na 85 324 (0.8–1.2 %) wpada poniżej najmniejszej znormalizowanej `float16` (6.1·10⁻⁵) i
  jest reprezentowane denormalnie, z obniżoną precyzją. Ich wartość bezwzględna jest 100–300× poniżej podłogi
  szumu, więc nie ma to znaczenia.

Błąd kwantyzacji jest zatem **dwa rzędy wielkości poniżej** szumu, którego i tak nie da się usunąć, a nawet
największy pojedynczy błąd (1.89·10⁻³) jest 4.3× mniejszy od najniższej zmierzonej podłogi szumu (0.0085).
Oszczędność: 21.4 GB → 10.7 GB.

**Reguła bezwzględna: uśredniać w `float32`, rzutować na `float16` dopiero na gotowym wyniku.** Akumulacja w
`float16` kumulowałaby błąd zaokrąglenia przy każdym z N dodawań i powyższe liczby przestałyby obowiązywać.

**Głębia zostaje `float32`** — celowo, mimo że `float16` zaoszczędziłoby ~4 GB. Przy zasięgu do 12.66 m
(`PKL_FORMAT.md`) `float16` daje błąd ~6 mm, czyli 250× więcej niż zweryfikowana wierność odtworzenia pkl
(RMSE głębi 2.4·10⁻⁵ m). Nie warto tracić bitowej porównywalności z pkl dla 4 GB. RGB jest `uint8` z definicji
formatu.

---

## 5. Ograniczenia do wypunktowania w pracy

1. **Echa pochodzą z innego silnika niż wszystkie opublikowane baseline'y** (SoundSpaces 2.0 on-the-fly vs 1.0
   prekomputowane RIR-y u Gao i Paridy). Metryki bezwzględne nie są porównywalne między silnikami. Wiarygodne
   porównanie to 36 vs 4 orientacje **wewnątrz naszego datasetu**, z baseline'em 4-orientacyjnym wygenerowanym
   z **naszych** renderów — nigdy odczytanym z tabeli Gao. Kąty 0/90/180/270 leżą w siatce co 10°, więc jedna
   generacja daje oba warunki.
2. **Obciążenie od liczby promieni zależy od orientacji, ale nieistotnie**: 2.1 % sygnału 10°
   (`e2_bias_orientation`). Struktura jest realna (harmoniczna k=2, okres 180°, ta sama w dwóch scenach), tylko
   ~47× mniejsza od mierzonego efektu.
3. **2.14 % powierzchni scen ma materiał domyślny** i nie da się tego naprawić configiem — 2.09 % to obiekty
   `class_id: -1` (null `category()`), najgorzej w `office_0` (11.5 %) i `office_4` (9.4 %).
   Patrz `REPLICA_MATERIALS.md §5`.
4. **Odtwarzalność bit-exact** wymaga `threadCount = 1`. Nie jest to koszt: `threadCount = T` przy
   `indirectRayCount = R` liczy **R/T promieni**, a przy równej jakości 8 wątków jest 1.18× **wolniejsze**
   (4000/8 = 0.3335 s vs 500/1 = 0.2823 s) — `e2_thread_budget_confirm`.
5. Liczby szumu z wczesnej charakteryzacji (E1–E4, checkpoint-boundary) mierzono przy słuchaczu na 1.5 m;
   produkcja idzie na 1.25 m. Wartości z Bloków 3 i `noise_floor_orientation` są już przy 1.25 m i zgadzają się
   z historycznymi w granicach rozrzutu.
6. **Próbki obcięte przez `N_MAX`.** Dla pozycji o szumie powyżej `sigma_1 = 0.1164` reguła adaptacyjna zażąda
   więcej niż 40 renderów i zostanie obcięta; takie próbki mogą nie osiągnąć SNR 3.5 i gwarancja jakości ich
   nie obejmuje. Identyfikacja w metadanych (§3.4.1) jest jednoznaczna:

   ```
   clamped == "max"  AND  snr_final < 3.5
   ```

   Raportować w pracy: ich liczbę, udział w 62 640 próbkach, rozkład `snr_final` oraz `n_raw` (o ile clamp
   zaniżył `N`). Oczekiwana liczba to zero — `N_MAX = 40` pokrywa cały udokumentowany zakres szumu
   (`CLAUDE.md`: do `sigma_1 = 0.1131`) — ale to musi być **stwierdzone pomiarem**, nie założone. Jeśli
   obcięte próbki się pojawią, można je uzupełnić bez powtarzania generacji: pętla weryfikacyjna z §3.4 już
   je oznacza, więc wystarczy dorenderować je z podniesionym `N_MAX`.
7. **Kwantyzacja do `float16`** wnosi błąd 7–8·10⁻⁵ RMSE do każdej zapisanej próbki (§4.1). Jest 112–240×
   poniżej podłogi szumu, ale nie jest zerem — przy analizach porównujących różnice rzędu 10⁻⁴ trzeba o nim
   pamiętać. Osiągnięty SNR (`snr_probe`, `snr_final`) liczyć **przed** rzutowaniem.

---

## 6. Otwarte, świadomie odłożone

1. **Config materiałów per scena** dla kategorii `floor` — podłogi drewniane w rodzinie apartamentów kontra
   szare w `office_0..4`, `room_0`, `room_2`. Kosztuje zero (i tak jeden Simulator na scenę), nieprzetestowane.
   `REPLICA_MATERIALS.md §6`.
2. **Walidacja wobec publicznych pomiarów rzeczywistych apartamentu FRL** (SoundSpaces 2.0, Sek. 5.2). To nasza
   rodzina scen; dałoby obiektywne kryterium doboru liczby promieni zamiast samoodniesienia do 5000.
3. E5 — mapa przestrzenna.
