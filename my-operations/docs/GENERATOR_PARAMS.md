# Parametry generatora datasetu — FINALNE

Specyfikacja zamrożona **2026-07-26**, po zamknięciu fazy charakteryzacji i walidacji.
To jest dokument referencyjny dla sesji, w której powstanie generator — **nie jest to kod generatora**.

Każda pozycja ma odwołanie do eksperymentu, który ją rozstrzygnął. Surowe wyniki:
`outputs/diagnose_rlr_noise_out/diagnostics_report.json` (klucz podany przy każdej pozycji).
Konteksty: `docs/PKL_FORMAT.md` (kamera i zbiór lokalizacji), `docs/REPLICA_MATERIALS.md` (materiały akustyczne).

---

## 1. Tabela parametrów

| parametr | wartość | źródło |
|---|---|---|
| `indirect_ray_count` | **500** | `e2_bias_orientation`, `e2_rays_vs_renders` |
| `thread_count` | **1** | `e2_thread_budget_confirm` |
| `n_renders` | **adaptacyjne per lokalizacja**, `N ∈ [6, 64]`, patrz §3 | `noise_floor_scenes`, `probe_census` |
| `averaging_domain` | **`"mag"`** = (1/N)Σ\|STFT\| | `e3_averaging_domain` |
| `material_config` | **`my-operations/replica_material_config.json`** | `materials_verify` |
| `listener_height` | **1.25 m** (kamera i audio) | `listener_height`, `PKL_FORMAT.md` |
| `simulator_rotation` | **nie dotyczy** — brak wycieku | `gpu_memory_scale` |
| `spectrogram_dtype` | **`float16`** na dysku, akumulacja w `float32` | §4.1 |
| `depth_dtype` | **`float32`** (celowo nie float16) | §4.1 |
| `audio_sims_per_render` | **1** (było 2 — zdublowana, nieodczytywana symulacja) | §4.3 |
| `warmup_discard` | **500** renderów odrzucanych po konstrukcji `Simulator` | §2 |
| `est_time_total` | **≈ 31.9 h** (z pełnego census, nie z ekstrapolacji) | §4 |
| `est_disk` | **≈ 14 GB** (zmierzone po gzip) | §4 |

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

WARMUP_DISCARD = 500   # renderow wykonanych i odrzuconych zaraz po konstrukcji
                       # Simulatora, przed pierwsza lokalizacja (patrz nizej)
```

Obserwacje wizualne zapisywać **surowe** z `get_sensor_observations()` — bez normalizacji i bez klipowania
głębi (`PKL_FORMAT.md`: pkl trzyma RGBA uint8 i głębię w surowych metrach).

### Rozgrzewka Simulatora (dodane 2026-07-29)

Pierwsze ~10 renderów w świeżej instancji `Simulator` ma systematycznie **wyższy szum**. Zmierzone
estymatorem wariancyjnym w blokach po 10 renderów, po 100 renderów na pozycję, na trzech scenach:

| pozycja | blok 1 (r0–9) | stan ustalony (r40–99) | nadwyżka |
|---|---|---|---|
| `office_1/33` | 0.11286 | 0.10130 ± 0.00266 | **+11.4 %** (+4.4 SD) |
| `frl_apartment_5/186` | 0.03945 | 0.03289 ± 0.00041 | **+19.9 %** (+16.1 SD) |
| `room_0/43` | 0.06984 | 0.06329 ± 0.00090 | **+10.4 %** (+7.3 SD) |

**Efekt jest własnością KONSTRUKCJI Simulatora, nie pozycji agenta.** Rozstrzygnięte bezpośrednio: po
przeniesieniu agenta na drugą pozycję **w tej samej instancji** (po 100 renderach) pierwszy blok już **nie**
jest podwyższony — stosunek „pierwszy blok / pozostałe" wynosi 1.004, 0.999 i 0.993, wobec 1.114, 1.199
i 1.104 na pozycji pierwszej. Skala problemu to więc **18 lokalizacji** (po jednej na scenę), a nie 1740.

Dotyczy **wyłącznie szumu, nie średniej**: energia spektrogramu w pierwszych 10 renderach różni się od stanu
ustalonego o −0.88 %, −0.77 % i −0.68 %, czyli 1.75, 1.36 i 1.14 SE — poniżej progu istotności, choć wszystkie
trzy tym samym znakiem. Odrzucenie rozgrzewki usuwa oba zastrzeżenia naraz.

Dlaczego to naprawiamy, skoro kierunek błędu jest „bezpieczny": sonda pierwszej lokalizacji każdej sceny
wypada w całości w okresie rozgrzewki, więc zawyża `sigma_1`, a przez `N ∝ sigma_1²` zawyża `N` o ~20–45 %.
Te lokalizacje dostałyby **więcej** renderów, niż trzeba, czyli SNR wyższy od reszty zbioru — jednorodność
szumu psułaby się dokładnie tak samo jak przy niedrzucaniu nadmiaru sondy dla orientacji 0° (§3.3 pkt 2),
i tak samo byłaby skorelowana z czymś, co nie ma nic wspólnego z badanym efektem.

**Wartość: `WARMUP_DISCARD = 500`** (rewizja 2026-07-29). Zmierzony punkt osiadania jest znacznie niższy —
w blokach po 5 renderów nadwyżka spada z +16…19 % (r0–4) poniżej rozrzutu blokowego już od r20, a estymata
skumulowana od renderu 20 do końca leży 0.35–0.75 % od odniesienia (r50–99). Wybrano jednak 500, bo:

- `gpu_memory_scale` udokumentował **500 renderów** jako koniec fazy rozgrzewki dla RSS, pamięci GPU i czasu
  renderu — spójność z tamtym pomiarem;
- to **25× zmierzony punkt osiadania**, więc żaden realny transient się w nim nie zmieści;
- kosztuje 500 × 0.1412 s = 71 s na scenę, czyli **21 min** na cały zbiór (+1.1 %) — cena znikoma wobec
  ryzyka systematycznego biasu.

Że 20 nie wystarczało, pokazał dopiero pełny census sondy (§3.2): pierwsza sondowana lokalizacja każdej sceny
wypadała systematycznie wyżej (mediana percentyla **92 %**, Wilcoxon **p = 0.001**). Domiar w stanie ustalonym
wykazał, że to w większości efekt **przestrzenny** — `loc_id` rośnie wzdłuż siatki punktów, więc id 0 to róg
sceny, często przy ścianach (`frl_apartment_3/0`: 0.09444 w stanie ustalonym przy medianie sceny 0.04550) —
ale rozgrzewka dokładała do tego kilka procent. Bezpośredni test: `office_1/5` daje `sigma_1` = 0.0918 (N=25)
przy `WARMUP_DISCARD = 20` i **0.0817 (N=20)** przy 500.

Wykres: `outputs/diagnose_rlr_noise_out/warmup_simulator.png`.

### Potok spektrogramu (bez zmian, `echo_core/spectrogram.py`)

```
SAMPLE_RATE = 44100 ;  ECHO_MS = 60  ->  ECHO_SAMPLES = 2646
STFT_N_FFT = 512 ;  STFT_WIN_LENGTH = 64 ;  STFT_HOP_LENGTH = 16
chirp: my-operations/sweep_audio/3ms_sweep.wav
spektrogram: (2, 257, 166)  -- float32 w pamieci, float16 na dysku (patrz 4.1)
```

### Zbiór próbek

- **Lokalizacje: 1740** — klucze `scene_observations_128.pkl`, nie `points.txt` w całości. To zbiór
  odpowiadający próbkowaniu z pracy Gao (VisualEchoes, ECCV 2020), więc tylko on daje porównywalność.
- Współrzędne z `points.txt`: `x = a`, `z = -b`; `y` — **z `graph.pkl`** (`node["point"][1]`, pełna precyzja
  `float32`), a dla 8 lokalizacji spoza grafu ta sama stała sceny (`PKL_FORMAT.md`, tabela wysokości).
  `location_id` w pkl **jest** kolumną `id` z `points.txt`, a współrzędne obu źródeł są identyczne co do bitu
  (`PKL_FORMAT.md`).

  > **Poprawka 2026-07-28.** Do tej daty stało tu `y` z `pathfinder.snap_point([x, y_guess, z])` — kalka
  > z `diagnose_rlr_noise.py:102 load_point_position()`, czyli z kodu diagnostyk, nie z `PKL_FORMAT.md`.
  > Jest to **błędne**: `snap_point()` zwraca wysokość powierzchni navmesha, a nie podłogi. Navmesh Repliki
  > nie ma zapisanych `NavMeshSettings`, więc recast odtwarza go z domyślną kwantyzacją i leży
  > **~0.21 m nad `y` z grafu** — mediana 0.2125 m, maksimum 0.4901 m, na **1738 z 1740** lokalizacji
  > (zmierzone na wszystkich 18 scenach). Nie zależy to od `y_guess`: podanie `y` z grafu jako punktu
  > startowego daje wynik co do bitu identyczny, bo `snap_point` rzutuje na navmesh niezależnie od startu.
  >
  > Rozstrzygnięcie jest pomiarowe, nie interpretacyjne. `office_1`, 16 lokalizacji × 4 kąty = 64 porównania
  > piksel-po-pikselu z `scene_observations_128.pkl` (rendering wizualny jest deterministyczny, więc test ma
  > moc rozstrzygającą — por. kontrola negatywna w `PKL_FORMAT.md`):
  >
  > | wariant `y` | RGB RMSE śr. / max | % pikseli bit-identycznych | depth RMSE śr. / max |
  > |---|---|---|---|
  > | `graph.pkl` | **0.0125 / 0.0214** | **99.982 %** | **9·10⁻⁶ / 2.2·10⁻⁵ m** |
  > | `snap_point` | 50.05 / 75.28 | 36.02 % | 0.150 / 0.423 m |
  >
  > Replikuje to wynik `PKL_FORMAT.md` (0.0077 / 99.99 % na `room_0`) na scenie tam nietestowanej.
  > Konsekwencja dla interpretacji pomiarów szumu — patrz §5 ograniczenie 8.
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

> Bezwzględne godziny w tej tabeli pochodzą sprzed zmiany z §4.3 (0.2606 s/render, stara ścieżka z podwójną
> symulacją) i **nie są aktualnym budżetem** — ten jest w §4. Stosunek 2.14× pozostaje ważny, bo obie kolumny
> skalują się tym samym czynnikiem; to on jest treścią tego porównania.

### 3.2 Kryterium

Wszystko liczone z renderów, które i tak powstają — **żaden dodatkowy pomiar wstępny nie jest potrzebny**.
Estymator szumu to ten sam podział na dwie rozłączne połówki, którego używa każdy eksperyment w tym projekcie:

```
sigma_1 = RMSE(polowka_A, polowka_B) / sqrt(2) * sqrt(n/2)     # szum POJEDYNCZEGO renderu
N       = clamp( ceil( (TARGET_SNR * sigma_1 / SIGNAL_10DEG)^2 ), N_MIN, N_MAX )

TARGET_SNR   = 3.5
SIGNAL_10DEG = 0.0644      # mediana z noise_floor_scenes, stała między scenami
N_MIN, N_MAX = 6, 64
```

`RMSE(A, B) = sqrt(2)·sigma_N` — stąd dzielenie przez `sqrt(2)`. Nigdy nie porównywać surowych RMSE dwóch
zaszumionych estymat bez tej dekompozycji (błąd popełniony raz w Bloku B).

#### Skąd `TARGET_SNR = 3.5` — uczciwie: to wybór projektowy, nie wielkość wyprowadzona

**Co ten próg znaczy:** `SIGNAL_10DEG` to RMSE między spektrogramami oddalonymi o 10°, czyli **najmniejsza
różnica, jaką zbiór ma w ogóle reprezentować** (krok siatki 36 orientacji). `sigma_N = sigma_1/sqrt(N)` to
resztkowy szum Monte Carlo po uśrednieniu. Warunek `SIGNAL_10DEG / sigma_N >= 3.5` mówi więc: *różnica
między sąsiednimi orientacjami ma być 3.5× większa niż szum renderowania*.

**Dlaczego próg musi być wyraźnie powyżej 1 — to jest twarda przesłanka.** Zmierzone `sigma_1` na
lokalizację: **0.0258–0.1327** w wariancie `main` i **0.0383–0.1506** w `patched`, przy sygnale 10°
równym 0.0644. Przy **pojedynczym** renderze szum w najgorszych lokalizacjach jest więc **2.06×
(`main`) / 2.34× (`patched`) większy od sygnału**, który ma być mierzony. Bez uśredniania sąsiednie
orientacje byłyby nierozróżnialne i zbiór 36-orientacyjny niósłby w dużej części szum zamiast
informacji kątowej. **Jakieś** uśrednianie jest zatem konieczne, a nie opcjonalne.

**Ale konkretna wartość 3.5 nie ma w tym repozytorium żadnego wyprowadzenia.** Dokument definiuje
względem niej wszystko inne (progi `N_MAX`, budżet, sformułowanie z §3.5), sama liczba jest jednak
przyjęta, a nie policzona. Przy pisaniu pracy należy ją opisać jako **wybór konserwatywny**, nie jako
wynik. Koszt jest kwadratowy (`N ~ SNR^2`) — zmierzone na gotowym zbiorze średnie `N`:

| `TARGET_SNR` | średnie `N` (main) | względem 3.5 |
|---|---|---|
| 2.0 | 6.35 | 0.52× |
| **3.5** | **12.13** | **1.00×** |
| 5.0 | 23.40 | 1.93× |

**Jak wyszło w praktyce** (`snr_final` po dorenderowaniu, oba warianty): mediana 3.72 / 3.66, a **79 % / 94 %
próbek leży w przedziale 3.5–4.0** i **0.0 % powyżej 6**. Reguła celuje dokładnie w próg i go nie
przekracza — to zamierzone (§4), ale znaczy też, że zbiór **nie ma zapasu**: obniżenie progu wymagałoby
regeneracji, a nie odrzucenia części renderów.

Zastrzeżenie do samego pomiaru: `snr_final` liczy ten sam estymator połówkowy, który ma sufit dokładności
4–6 % (§2.6 raportu). „≥ 3.5" znaczy więc „≥ 3.5 według estymatora o ~5 % rozrzutu".

#### Dlaczego `N_MAX = 40`, a nie 24 (rewizja 2026-07-26 — HISTORYCZNE, patrz niżej)

Pierwotne `N_MAX = 24` odpowiada progowi `sigma_1 = sqrt(24)·0.0644/3.5 = 0.09014`. Przegląd wszystkich
dotychczasowych pomiarów pokazał, że ten próg jest ustawiony **dokładnie na krawędzi zmierzonych danych**:

- z 12 pozycji o zmierzonym szumie (`noise_floor_scenes` — 8 pozycji przy 1.25 m; `signal_noise_recheck`
  z configiem `replica` — 4 pozycje przy 1.5 m) **żadna** nie przekracza `N_raw = 24`. Rozkład `N_raw`:
  **3, 6, 7, 7, 9, 9, 9, 10, 11, 12, 13, 19** (przed podmianą dwóch pozycji na mediany z pełnego obrotu, patrz §3.1);
- ale na poziomie pojedynczych orientacji (`noise_floor_orientation`, 72 pomiary) najgorsza wypada przy
  `sigma_1 = 0.09006`, czyli **0.09 % poniżej progu obcięcia**. Margines jest zerowy.

Rozstrzygający jest jednak argument spoza tej próbki: charakteryzacja z 07-20 dokumentuje
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

#### Potwierdzenie na próbce 52 pozycji (2026-07-29 — HISTORYCZNE, zastąpione przez census niżej)

Powyższe uzasadnienie opierało się na 12 pozycjach z 4 scen i na udokumentowanym zakresie szumu, a **11 z 18
scen (1227 z 1740 lokalizacji, 71 %) nie miało żadnego pomiaru**. Ponieważ pierwsza próbka przy limicie
pojawiła się realnie (`office_1`, §5 ogr. 6), próg został zweryfikowany bezpośrednio: eksperyment
`noise_floor_remaining` zmierzył po 2 pozycje (ułamki 0.20 i 0.75 listy lokalizacji) we wszystkich 11
brakujących scenach, w **konfiguracji dokładnie produkcyjnej** — `y` z `graph.pkl`, sensor 1.25 m, 500 promieni,
1 wątek, materiały Repliki, 1 symulacja audio na render, `WARMUP_DISCARD = 20` — estymatorem wariancyjnym,
M = 20 renderów na pozycję, 11 konstrukcji `Simulator`.

Mediana `sigma_1` per scena, wszystkie 18 scen, 52 pozycje łącznie (posortowane malejąco):

| scena | mediana `sigma_1` | `N` z reguły | zapas do progu 0.11637 |
|---|---|---|---|
| `hotel_0` | 0.07961 | 19 | 1.46× |
| `office_1` | 0.07659 | 18 | 1.52× |
| `apartment_0` | 0.07340 | 16 | 1.59× |
| `office_3` | 0.07334 | 16 | 1.59× |
| `room_1` | 0.07223 | 16 | 1.61× |
| `frl_apartment_3` | 0.07118 | 15 | 1.63× |
| `room_2` | 0.06869 | 14 | 1.69× |
| `office_2` | 0.06512 | 13 | 1.79× |
| `frl_apartment_2` | 0.06489 | 13 | 1.79× |
| `office_0` | 0.06343 | 12 | 1.83× |
| `apartment_1` | 0.06318 | 12 | 1.84× |
| `office_4` | 0.05860 | 11 | 1.99× |
| `apartment_2` | 0.05567 | 10 | 2.09× |
| `room_0` | 0.05313 | 9 | 2.19× |
| `frl_apartment_4` | 0.04421 | 6 | 2.63× |
| `frl_apartment_1` | 0.04185 | 6 | 2.78× |
| `frl_apartment_0` | 0.04155 | 6 | 2.80× |
| `frl_apartment_5` | 0.03387 | 4 | 3.44× |

**Wynik: `N_MAX = 40` wystarcza.**

- **0 z 52** zmierzonych pozycji przekracza `N_MAX`; **3 z 52** przekraczają porzucony próg 24 — co
  potwierdza, że rewizja `N_MAX` 24 → 40 była konieczna, a nie ostrożnościowa.
- Najwyższe zaobserwowane `sigma_1` = **0.10091** (`office_1`, pełny pomiar 16 lokalizacji) → `N_raw` = 31,
  zapas do progu **1.15×**. Wśród 22 nowych pozycji maksimum to 0.09113 (`frl_apartment_3`) → `N_raw` = 25.
- Żadna scena nie ma mediany `sigma_1` powyżej 0.10. Najgłośniejszy jest `hotel_0` (0.07961), nieznacznie
  przed `office_1` (0.07659) — różnica mieści się w niepewności estymaty z 3 pozycji.

Przewidywany odsetek próbek, które dobiją do `N_MAX` i **nie** osiągną progu, liczony przez zastosowanie
empirycznego rozkładu względnego `sigma_1` wewnątrz sceny (z 576 próbek `office_1`, `sigma` per próbka
z nieocenzurowanego `snr_probe`; p95 = 1.263, p99 = 1.387, max = 1.734) do mediany każdej sceny:

| scena | % próbek przy limicie | sztuk |
|---|---|---|
| `hotel_0` | 0.52 % | 9.0 |
| `apartment_0` | 0.17 % | 13.2 |
| `frl_apartment_3` | 0.17 % | 9.2 |
| `office_3` | 0.17 % | 3.9 |
| `room_2` | 0.17 % | 2.9 |
| `room_1` | 0.17 % | 2.2 |
| `office_1` | 0.17 % | 1.0 |
| pozostałe 11 scen | 0.00 % | 0 |
| **razem** | **0.066 %** | **≈ 41 z 62 640** |

Model jest skalibrowany: dla `office_1` przewiduje 0.17 %, a zaobserwowano dokładnie 1 próbkę na 576 = 0.17 %.

Ekstrapolacja parametryczna na poziomie *lokalizacji* (dopasowanie do 52 zmierzonych `sigma_1`) daje 1.79 %
przy rozkładzie lognormalnym i 0.14 % przy normalnym. **Żadnej z tych liczb nie należy traktować jako
twierdzenia** — próbka jest mała i **nie dobrana losowo** (pozycje wybrane po ustalonych ułamkach listy),
a zaobserwowano 0 przekroczeń na 52. Podana wyżej liczba 0.066 % jest lepiej ugruntowana, bo jedyny model
kalibrowany na pełnej scenie.

#### `N_MAX = 64` — rozstrzygnięte pełnym census sondy (2026-07-29)

Wszystkie dotychczasowe uzasadnienia progu opierały się na **próbce** pozycji (12, potem 52), dobranej po
ustalonych ułamkach listy lokalizacji. Zamiast ekstrapolować dalej, policzono **rzeczywisty rozkład dla
wszystkich 1740 lokalizacji**: sonda 8-renderowa jest i tak pierwszym krokiem generatora, więc da się ją
wykonać osobno bez renderowania 36 orientacji. Koszt: 14 300 renderów, **35 min**
(`generate_echo_dataset.py --probe-only`, wyniki w `outputs/probe_census/`).

`sigma_1`: mediana 0.05830, zakres **0.02530–0.13451**. `N_raw`: mediana 11, średnia 11.80, zakres **2–54**.

| `N_raw` | lokalizacji | udział |
|---|---|---|
| 1–5 | 319 | 18.33 % |
| 6–8 | 333 | 19.14 % |
| 9–12 | 436 | 25.06 % |
| 13–16 | 314 | 18.05 % |
| 17–20 | 149 | 8.56 % |
| 21–24 | 85 | 4.89 % |
| 25–30 | 69 | 3.97 % |
| 31–40 | 28 | 1.61 % |
| 41–48 | 6 | 0.34 % |
| 49–64 | 1 | 0.06 % |
| **> 64** | **0** | **0.00 %** |

**Ekstrapolacja z 52 pozycji chybiła.** Przewidywała zero przekroczeń `N_MAX = 40`; census znalazł **7
lokalizacji (0.402 %)**. Powód jest pouczający: gorące miejsce akustyczne w `apartment_0` siedzi przy
`loc_id` 285–310, czyli w okolicy ułamka 0.9 listy, a próbkowano ułamki 0.20 i 0.75. To dokładnie ten rodzaj
błędu, przed którym zastrzegano („próbka nie dobrana losowo") — tylko że tym razem się zmaterializował.

Wszystkie 7 przekroczeń domierzono estymatorem wariancyjnym (M = 40, SD ~1 %), **głęboko w stanie ustalonym**:

| scena | lok | census (sonda n=8) | domiar (M=40) | `N` | |
|---|---|---|---|---|---|
| `frl_apartment_2` | 0 | 0.13451 | **0.10703** | 54 → 34 | artefakt sondy |
| `apartment_0` | 285 | 0.12432 | **0.12799** | 46 → **49** | potwierdzony, **maksimum** |
| `apartment_0` | 307 | 0.12652 | 0.12632 | 48 → **48** | potwierdzony |
| `apartment_0` | 308 | 0.12390 | 0.12449 | 46 → **46** | potwierdzony |
| `apartment_0` | 310 | 0.12030 | 0.11799 | 43 → **42** | potwierdzony |
| `hotel_0` | 101 | 0.12136 | 0.11305 | 44 → 38 | spadł poniżej |
| `hotel_0` | 95 | 0.11693 | 0.10959 | 41 → 36 | spadł poniżej |

Cztery potwierdzone przekroczenia, wszystkie w `apartment_0` i w sąsiadujących `loc_id` — jedno realne
gorące miejsce, nie rozproszony szum. **Prawdziwe maksimum: `sigma_1` = 0.12799 → `N_raw` = 49.**

| `N_MAX` | pokrywa `sigma_1` do | lokalizacji obciętych | koszt |
|---|---|---|---|
| 40 (poprzednie) | 0.11637 | 7 | — |
| 48 | 0.12748 | 1 | +0.05 h |
| **64 (przyjęte)** | **0.14720** | **0** | **+0.06 h** |

`N_MAX = 48` odpadło, bo pokrywa do 0.12748 — **poniżej** potwierdzonego maksimum 0.12799, czyli obcinałoby
dokładnie tę lokalizację, którą zmierzono najdokładniej. Margines jest potrzebny również dlatego, że
produkcyjna sonda zmierzy każdą lokalizację **na nowo, raz, z SD ~5 %**: lokalizacja z census `N_raw` = 38
może w produkcji wypaść 45. Przy `N_MAX = 64` **żadna z 1740 lokalizacji nie jest obcinana**, a koszt to
3.6 min na 31.9 h (0.2 %).

Wykres: `outputs/diagnose_rlr_noise_out/probe_census.png`.

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

To jest lepsze niż ślepe zawyżanie `N` „na wszelki wypadek": oszacowanie `sigma_1` z sondy jest niedokładne,
a margines pokrywający tę niedokładność zjadłby przewagę czasową. Weryfikacja po fakcie jest dokładna
i kosztuje tylko dorenderowanie części przypadków.

#### Zmierzona dokładność sondy (2026-07-29)

Wcześniejsza wersja tego akapitu podawała „~10 % błędu `sigma_1`, ~20 % błędu `N`" — liczby **oszacowane,
nigdy niezmierzone**. Pomiar (bootstrap 4000×, 80 renderów stanu ustalonego na pozycję, referencja
z estymatora wariancyjnego po wszystkich 80):

| pozycja | `sigma_1` ref. | SD sondy n=8 | obciążenie | percentyle 5/50/95 | SD `N` |
|---|---|---|---|---|---|
| `office_1/33` | 0.10050 | **5.5 %** | −0.1 % | 0.917 / 0.994 / 1.093 | **11.1 %** |
| `frl_apartment_5/186` | 0.03312 | **4.2 %** | −0.1 % | 0.937 / 0.996 / 1.077 | **8.5 %** |
| `room_0/43` | 0.06348 | **4.5 %** | −0.1 % | 0.934 / 0.996 / 1.076 | **9.0 %** |

Sonda jest więc **dwukrotnie dokładniejsza, niż zakładano** (4–6 %, nie 10 %), i praktycznie nieobciążona.

**Zaskakujące i warte odnotowania: dokładność estymatora połówkowego nie zależy od `n`.** Zmierzone SD wynosi
5.5 % / 4.2 % / 4.5 % jednakowo przy `n` = 8, 20 i 40. Powód jest strukturalny: estymator liczy RMSE po
~85 000 komórkach spektrogramu, a `RMSE²·h/2` jest estymatorem `σ²` o **jednym stopniu swobody na komórkę,
niezależnie od `h`**. Zwiększanie liczby renderów zmniejsza amplitudę różnicy `A−B`, ale nie jej względną
precyzję — tę ogranicza efektywna liczba niezależnych komórek (rzędu 600–1200 z 85 324, czyli silna korelacja
przestrzenna). Estymator wariancyjny (`σ² = średnia po komórkach z Var po renderach`) ma `n−1` stopni swobody
na komórkę i **poprawia się z `n`** — jego niepewność przy 80 renderach to 0.1–1.1 %.

Praktyczny wniosek: do porównań dokładniejszych niż ~5 % nie używać estymatora połówkowego, niezależnie od
tego, ile renderów mu się poda. Generator liczy nim dalej, bo `snr_probe`/`snr_final` muszą być spójne
z regułą wyznaczającą `N` — nie dlatego, że jest najlepszy.

#### Dlaczego dorenderowanie dotyczy ~40 % próbek, a nie „nielicznych"

Na `office_1` dorenderowania wymagało **41.0 %** próbek. To nie jest usterka ani objaw złej sondy — to
konsekwencja tego, że reguła celuje **dokładnie** w próg `SNR = 3.5`. Symulacja Monte Carlo całej reguły
odtworzona na prawdziwych renderach (4000 powtórzeń: losuj 8 renderów → `N` → losuj `N` renderów →
`snr_probe`) przewiduje:

| pozycja | mediana `N` | przewidywany odsetek dorenderowań |
|---|---|---|
| `office_1/33` | 30 | **46.2 %** (zaobserwowane na pełnej scenie: **41.0 %**) |
| `room_0/43` | 12 | 37.5 % |
| `frl_apartment_5/186` | 6 (clamp `N_MIN`) | **0.0 %** |

Mechanizm: `snr_probe < 3.5` zachodzi w przybliżeniu wtedy, gdy `sigma_1` zmierzone z `N` renderów wypadnie
**wyżej** niż `sigma_1` zmierzone z sondy. Oba estymatory są nieobciążone i mają podobny rozrzut, więc
zdarza się to w około połowie przypadków **z konstrukcji**. Odsetek spada do zera tam, gdzie `N` zostaje
obcięte przez `N_MIN` (scena na tyle cicha, że 6 renderów mocno przekracza próg) — stąd zależność od sceny.

Zgodność 46.2 % przewidywane wobec 41.0 % zaobserwowanych (przewidywanie liczone na jednej, najgłośniejszej
pozycji sceny) potwierdza, że reguła zachowuje się dokładnie tak, jak wynika z jej konstrukcji.

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
  zawiesza się sprzętowo. Odzysk wymaga prawdziwego resetu PCI (`echo 1 > /sys/bus/pci/devices/<id>/reset`
  po wyładowaniu modułów `nvidia*`); `nvidia-smi -r` ani remove+rescan nie wystarczają.
- **Bez rotacji instancji w obrębie sceny.** `gpu_memory_scale`: 3000 renderów w jednej instancji, RSS
  dokładnie płaskie (1268 MiB, +0.0 MiB/1000), GPU bez trendu (892–984 MiB przy rozrzucie pomiaru 19.5 MiB),
  czas renderu płaski ~0.29 s. Największa scena to `apartment_0` ≈ 76 tys. renderów w jednej instancji.
- **Checkpoint na granicy próbki jest BEZPIECZNY** (`e1_checkpoint_boundary_merge`, R=16: mediana |r| 0.0222
  vs 0.0205, przekroczenia 22/128 vs 22/128, Wilcoxon p=0.949). Wznawianie po awarii nie koreluje szumu.

Tempo: **0.1412 s/render** (zmierzone 2026-07-29 na **pełnej** scenie `office_1`, 11 957 renderów, po
usunięciu zdublowanej symulacji audio — §4.3; mikrobenchmark na dwóch lokalizacjach dawał 0.1456). Poprzednia wartość **0.2606 s/render** — średnia ważona liczbą próbek po 18 scenach — dotyczy starej
ścieżki i pozostaje właściwym odniesieniem dla wszystkich pomiarów charakteryzacji. Rozmiar sceny prawie nie
wpływa: cały rozrzut to 1.6× (`apartment_0` 0.3531 s, `frl_apartment_5` 0.2205 s — wartości sprzed zmiany).

### Budżet (z pełnego census, 2026-07-29)

Poprzednie szacunki opierały się na próbce pozycji. Po census (§3.2) znany jest **rzeczywisty rozkład `N`
dla wszystkich 1740 lokalizacji**, więc budżet nie jest już projekcją z przedziałem, tylko sumą:

```
srednie N po clamp [6, 64]     12.11
renderow probek + narzut       803 273   (narzut petli weryfikacyjnej 1.0578x)
renderow rozgrzewki             9 000   (18 scen x 500, czyli 21.2 min)
RAZEM                          812 273

CZAS  = 812 273 x 0.1412 s  =  31.9 h
DYSK  = 62 640 probek x 234.8 KiB (zmierzone po gzip -4)  =  14.0 GiB
```

Wszystkie składniki zmierzone, żaden założony: tempo i narzut z pełnej sceny `office_1` wygenerowanej nową
ścieżką, `N` z census wszystkich lokalizacji, rozgrzewka z `WARMUP_DISCARD`.

Dla porównania: przed usunięciem zdublowanej symulacji audio (§4.3) ten sam zbiór kosztowałby ~63 h.

Najdłuższe pojedyncze sceny (policzone z census, `N` po clamp):

| scena | lokalizacji | średnie `N` | czas |
|---|---|---|---|
| `apartment_0` | 211 | 20.32 | **6.42 h** |
| `apartment_1` | 176 | 11.56 | 3.06 h |
| `apartment_2` | 142 | 12.45 | 2.66 h |
| `frl_apartment_5` | 148 | 8.93 | 2.00 h |

Trzy sceny held-out (`apartment_2`, `frl_apartment_5`, `office_4`) łącznie **6.16 h** — komplet danych
testowych jest dostępny po ok. sześciu godzinach.

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

### 4.2 Kolejność generowania scen (dopisane 2026-07-28)

Sceny idą **jedna po drugiej, każda w osobnym procesie OS**, uruchamiane ręcznie. Kolejność nie jest
alfabetyczna ani losowa — jest **harmonogramowa**:

1. **`office_1`** (16 lok.) — scena walidacyjna generatora. Najmniejsza, ~25 min, robiona pierwsza po to,
   żeby `--verify` przeszło na kompletnym pliku, zanim ruszy cokolwiek długiego.
2. **Trzy sceny held-out**: `apartment_2` (142), `frl_apartment_5` (148), `office_4` (76). Mając komplet
   held-out można zacząć budować dataloader i pipeline treningowy, kiedy pozostałe sceny jeszcze się mielą —
   przy ~45 h generacji to około tygodnia różnicy w harmonogramie. To także sceny, z których pochodzą finalne
   liczby pracy, więc jeśli cokolwiek w formacie okaże się złe, chcemy to wiedzieć na nich najwcześniej.
3. **Trzy treningowe, po jednej z każdej rodziny**: `room_0` (57, charakteryzowana od początku projektu),
   `office_0` (26, najgorsze pokrycie materiałowe — 11.5 % powierzchni `class_id: -1`, `REPLICA_MATERIALS.md §5`),
   `hotel_0` (48, osobna kategoria sceny, czwarta scena Bloku 3). Razem 131 lokalizacji ≈ 3.3 h — po nich
   dataloader ma już dane treningowe ze wszystkich trzech rodzin scen.
4. **Reszta rosnąco po liczbie lokalizacji**: `room_1` (35), `office_2` (37), `room_2` (47), `office_3` (62),
   `frl_apartment_2` (125), `frl_apartment_4` (127), `frl_apartment_0` (130), `frl_apartment_1` (130),
   `frl_apartment_3` (147), `apartment_1` (176), `apartment_0` (211). Rosnąco, bo wtedy liczba **ukończonych**
   scen rośnie szybciej — a jednostką użyteczności jest gotowa scena, nie gotowa próbka.

Razem 1740 lokalizacji. `--status` wypisuje sceny w tej kolejności.

**Weryfikacja kolejności po pomiarze szumu (2026-07-29).** Powodem, dla którego warto by przestawić kolejność,
byłaby scena wyraźnie głośniejsza od reszty — zostawiona na koniec byłaby niespodzianką harmonogramową.
Pomiar `noise_floor_remaining` (§3.2) wyklucza taki przypadek: mediany `sigma_1` wszystkich 18 scen mieszczą się
w przedziale 0.034–0.080, czyli `N` od 4 do 19, bez wartości odstających. **Kolejność zostaje bez zmian.**

Dla przewidywalności warto jednak odnotować, ile trwają najdłuższe sceny przy projekcji centralnej:

| scena | lokalizacji | `N` | czas | pozycja w kolejności |
|---|---|---|---|---|
| `apartment_0` | 211 | 16 | **≈ 5.0 h** | 18 (ostatnia) |
| `frl_apartment_3` | 147 | 15 | ≈ 3.3 h | 16 |
| `apartment_1` | 176 | 12 | ≈ 3.2 h | 17 |
| `frl_apartment_5` | 148 | 4 | ≈ 1.0 h | 3 (held-out) |

`apartment_0` jest jednocześnie największa i trzecia pod względem szumu, więc to najdłuższy pojedynczy przebieg
w całym zbiorze — ale jest już **zmierzona**, więc nie jest ryzykiem, tylko znaną pozycją harmonogramu.
Trzy sceny held-out (pozycje 2–4) zajmą łącznie **≈ 4.3 h** (`apartment_2` 2.12 h, `frl_apartment_5` 0.91 h,
`office_4` 1.25 h) — komplet danych testowych jest więc dostępny po niespełna pięciu godzinach, co jest
istotą argumentu o wczesnym starcie dataloadera.

### 4.3 Jedna symulacja akustyczna na render zamiast dwóch (zmiana 2026-07-28)

**Co było źle.** `phase3_echolocation()` (dziś `echo_core/audio.py`) wywoływało
`audio_sensor.runSimulation(sim)` jawnie, a następnie `sim.get_sensor_observations()`. To drugie dla sensora
typu `AUDIO` wchodzi w `Sensor._get_audio_observation()`
(`habitat-sim/src_python/habitat_sim/simulator.py:763-777`), które **samo** ustawia transform słuchacza,
wywołuje `runSimulation()` **po raz drugi** i dopiero jego wynik zwraca przez `getIR()`. Jawne wywołanie
liczyło więc pełną symulację Monte Carlo, której wyniku nikt nie odczytywał. `AudioSensor::runSimulation()`
nie ma cache'u — flagi `newInitialization_`/`newSource_` sterują tylko uploadem geometrii i źródła, sama
symulacja idzie bezwarunkowo (`AudioSensor.cpp:164`).

Zmierzone: **283.8 ms → 143.2 ms** na `office_1`, **217.0 ms → 108.4 ms** na `frl_apartment_5` — dokładnie 2×.

**Co usunięto.** Wyłącznie jawne `runSimulation()`. **`setAudioSourceTransform()` zostaje** —
`_get_audio_observation()` ustawia wyłącznie transform *słuchacza*, więc bez tamtej linii źródło dźwięku nigdy
nie zostałoby ustawione i echolokacja (źródło współlokowane z odbiornikiem) przestałaby działać. Ustawienie
pozy agenta, materiałów i transformu słuchacza również zostaje. `phase3_echolocation()` przyjmuje teraz
`run_simulation=True` domyślnie, żeby `diagnose_rlr_noise.py` odtwarzał swoje historyczne liczby co do bitu;
generator podaje jawnie `False`, a plik HDF5 zapisuje to w atrybucie `audio_sims_per_render`.

**Dlaczego wymagało to walidacji.** Zmiana przesuwa sekwencję RNG: ścieżka podwójna zużywała 2 losowania na
render i używała co drugiego (#2, #4, #6…), pojedyncza zużywa 1 i używa każdego. Jest też różnica
mechanistyczna, nie tylko sekwencyjna — w ścieżce podwójnej obserwowana symulacja biegła z
`newSource_ == false`, w pojedynczej z `true` (źródło jest ponownie dodawane). Cała charakterystyka szumu
(`SIGNAL_10DEG`, rozkład `N`, 0.2606 s/render) była skalibrowana na ścieżce podwójnej.

**Pomiar równoważności (2026-07-28).** Dwie pozycje pokrywające zmierzony zakres szumu: `office_1/33`
(najgłośniejsza, `sigma_1 ≈ 0.10`) i `frl_apartment_5/186` (najcichsza, `sigma_1 ≈ 0.034`). Dla każdej:
kąty 0° i 10°, M = 40 renderów na kąt, ścieżki **nieprzeplatane** (komplet jedną instancją Simulatora, potem
komplet drugą). Niepewność z replikatów — każdy kąt dzielony na dwie rozłączne połówki po 20 renderów.

| pozycja | metryka | podwójna | pojedyncza | różnica | wynik |
|---|---|---|---|---|---|
| `office_1/33` | energia spektrogramu | 0.220816 | 0.221185 | +0.167 % | **0.72 SE**, Mann-Whitney p = 0.462 |
| | sygnał 10° | 0.06650 | 0.06538 | −1.7 % | **1.28 SE** |
| | `sigma_1` | 0.10048 | 0.10181 | +1.3 % | **0.51 SE** |
| `frl_apartment_5/186` | energia spektrogramu | 0.091763 | 0.091768 | +0.005 % | **0.02 SE**, p = 0.877 |
| | sygnał 10° | 0.06424 | 0.06426 | +0.0 % | **0.07 SE** |
| | `sigma_1` | 0.03375 | 0.03476 | +3.0 % | **0.75 SE** |

Wszystkie sześć porównań poniżej 2 SE (maksimum 1.28 SE). Oba pomiary sygnału 10° mieszczą się
w udokumentowanym zakresie 0.0639–0.0662, więc **`SIGNAL_10DEG = 0.0644` pozostaje ważne**. Wpływ na `N`
z reguły: 30 → 31 i 4 → 4. **Werdykt: RÓWNOWAŻNE.**

**Czego pomiar nie wyklucza.** Przy progu 2 SE czułość wynosi: energia ~0.2–0.5 %, sygnał 10° ~1.3–2.6 %,
`sigma_1` ~5–8 %. Różnicy `sigma_1` mniejszej niż ~5 % nie da się tą próbą wykryć. Nawet gdyby była realna,
oznaczałaby ~10 % więcej renderów, czyli przyspieszenie netto 1.8× zamiast 2.0× — i tak byłaby automatycznie
skompensowana, bo reguła adaptacyjna mierzy `sigma_1` **na miejscu**, a `snr_final ≥ 3.5` jest weryfikowane
po fakcie per próbka (§3.4). Gwarancja jakości zbioru nie zależy więc od tej różnicy.

**Uwaga metodologiczna o estymatorze.** Pierwsze podejście, estymatorem połówkowym z §3.2, dało pozornie
niepokojące +7.2 % i +18.4 % na `sigma_1`. Okazało się to artefaktem: estymator połówkowy ma na 40 renderach
SD rzędu 11–17 %, a dodatkowo jest skrajnie czuły na **rozgrzewkę** — pierwsze ~10 renderów po konstrukcji
`Simulator` jest wyraźnie głośniejsze (`office_1`: 0.10628 → 0.09930 → 0.09847 → 0.09814 w blokach po 10),
i to, w której połówce wylądują, przesuwa wynik o kilkanaście procent. Ta sama wielkość policzona wariancją
po wszystkich renderach (`sigma_1² = średnia po komórkach z Var_po_renderach`, M−1 stopni swobody zamiast ~1)
daje wartości w tabeli wyżej. **Do porównań o dokładności lepszej niż ~10 % nie używać estymatora
połówkowego** — generator liczy nim dalej, ale tam chodzi o zgodność z regułą na `N`, a nie o maksymalną
precyzję.

**Konsekwencja dla `office_1`.** Scena została wygenerowana **starą ścieżką** (`audio_sims_per_render = 2`),
zanim zmianę wprowadzono. Pozostałe 17 scen pójdzie nową. Rozkłady są równoważne (wyżej), a wartość jest
zapisana w atrybutach każdego pliku, więc różnica jest jawna i wykrywalna — ale dla pełnej jednorodności
zbioru `office_1` warto wygenerować od nowa (`--force`, ok. 35 min).

### 4.4 Układ katalogu wyjściowego (dodane 2026-07-29)

Każda scena dostaje własny podkatalog:

```
outputs/echoes_36deg/
    <scena>/
        <scena>.h5          dataset
        generate.log        log czytelny
        decisions.jsonl     jedna linia na lokalizację (sigma_1, n_raw, n_planned, clamped, czas)
        progress.json       sidecar dla --status (zapisywany atomowo, tmp + rename)
        verify/             PNG-i z --verify, nazwy generyczne locNN_angMMM.png
    .scene_index.json       cache liczby lokalizacji per scena (dla echo_ctl.py)
diagnose_rlr_noise_out/     bez zmian, WERSJONOWANY
```

Płaska struktura przy 18 scenach dawałaby kilkadziesiąt plików w jednym katalogu — łatwo pomylić scenę,
trudno skopiować albo skasować jedną bez ryzyka.

**Nazwa pliku HDF5 celowo powtarza nazwę katalogu** (`office_1/office_1.h5`, nie `office_1/echoes.h5`).
Ten jeden plik opuszcza katalog sceny — trafia na maszynę treningową i do kopii zapasowych — więc sama jego
nazwa musi mówić, co to jest. Pozostałe pliki mają nazwy generyczne, bo nigdy nie są oglądane poza swoim
katalogiem.

Wszystkie ścieżki wyjściowe wyprowadzane są z jednej funkcji `scene_dir()` w `generate_echo_dataset.py`
(oraz pochodnych `scene_h5`, `scene_log`, `scene_decisions`, `scene_progress`, `scene_verify_dir`,
`scene_stdout`); `echo_ctl.py` importuje je, nie skleja własnych. Kolejna zmiana układu wymaga edycji
w jednym miejscu.

Artefakty poprzedniej wersji sceny (np. `office_1` wygenerowany starą ścieżką z podwójną symulacją) zostają
**w katalogu swojej sceny** z sufiksem opisującym różnicę: `office_1_2sims.h5`, `generate_2sims.log`,
`decisions_2sims.jsonl`, `verify_2sims/`.

### 4.5 Dwa warianty datasetu: `main` i `patched` (dodane 2026-07-30)

Generator i pulpit przyjmują `--variant`. Warianty różnią się **wyłącznie geometrią sceny** —
wszystkie parametry z §1, reguła adaptacyjnego `N`, potok spektrogramu i zbiór lokalizacji są
identyczne.

| | `main` (domyślny) | `patched` |
|---|---|---|
| geometria | oryginalna Replica | dziury domknięte (`measurements/patch_scene_holes.py`) |
| scen | **18** | **10** |
| lokalizacji | 1740 | 1224 |
| próbek | 62 640 | 44 064 |
| katalog | `outputs/echoes_36deg/` | `outputs/echoes_36deg_patched/` |
| porównywalność z VisualEchoes | **tak** (RGB/depth 99.98 % pikseli bit-identycznych) | nie |

**Dlaczego `patched` ma tylko 10 scen.** Dziurę miało dokładnie 10 scen; pozostałe 8 jest
akustycznie szczelnych (§5 ogr. 10), więc nie ma w nich czego domykać i ich siatka jest
w obu wariantach **tym samym plikiem**. Przy `threadCount = 1` dałyby echa bit-identyczne,
więc generowanie ich po raz drugi to czysta strata czasu GPU. **Do treningu wariant dodatkowy
składa się z tych 10 scen plus 8 scen szczelnych z wariantu głównego.** `echo_ctl.py` odmawia
uruchomienia sceny szczelnej w wariancie `patched` i mówi dlaczego.

**Rozdzielenie wariantów.** Osobne katalogi wyjściowe, osobny cache indeksu scen, osobne
statusy. Każdy plik HDF5 ma atrybut **`variant`**, a `scene_id` wskazuje faktycznie użytą
siatkę — pliki obu wariantów mają te same nazwy i wymiary, więc bez tego atrybutu po
skopiowaniu na maszynę treningową nie dałoby się ich rozróżnić. Pulpit ostrzega, jeśli
generacja działa w innym wariancie niż oglądany (GPU jest jedno).

**Po co wariant dodatkowy.** Żeby sprawdzić empirycznie, czy domknięcie scen poprawia uczenie
predykcji głębi z echa. Uzasadnienie fizyczne łaty jest zmierzone i **niezależne od
jakiegokolwiek baseline'u** (§5 ogr. 10, `RAPORT_SESJI` §2.13–§2.15): ucieczka promieni
22 % → 0.00 %, zgodność z Eyringiem 0.41× → 1.00×, wysokość pomieszczenia zgodna ze scenami
szczelnymi. **Zastrzeżenie, które musi iść razem z każdym wynikiem z tego wariantu:** sceny
załatane są mierzalnie **bardziej wyidealizowane** niż nienaruszone skany (rozdzielenie
zupełne wobec scen szczelnych, Mann-Whitney p = 0.0032), bo łata jest idealnie płaską,
jednorodną płaszczyzną z jednego materiału — dokładnie tym, co zakłada model dyfuzyjny.

**Wariant `patched` wymaga wcześniejszego wygenerowania siatek** (742 MB, poza gitem):

```
python my-operations/measurements/patch_scene_holes.py --all
```

---

## 5. Ograniczenia do wypunktowania w pracy

1. **Echa pochodzą z innego silnika niż wszystkie opublikowane baseline'y** (SoundSpaces 2.0 on-the-fly vs 1.0
   prekomputowane RIR-y u Gao i Paridy). Metryki bezwzględne nie są porównywalne między silnikami. Wiarygodne
   porównanie to 36 vs 4 orientacje **wewnątrz naszego datasetu**, z baseline'em 4-orientacyjnym wygenerowanym
   z **naszych** renderów — nigdy odczytanym z tabeli Gao. Kąty 0/90/180/270 leżą w siatce co 10°, więc jedna
   generacja daje oba warunki.

   **Zmierzone 2026-07-30 (`measurements/cross_engine_rt60.py`), nie już tylko ostrożność.** Na **6 scenach
   Replica o całej geometrii** (gdzie nie ma czego łatać ani dobierać, więc pomiar nie ma wolnych parametrów)
   RT60 z naszego SS 2.0 wobec prekomputowanych RIR-ów SS 1.0 daje:

   | | wartość |
   |---|---|
   | średnia geometryczna | **0.95×** — silniki są nieobciążone względem siebie |
   | geometryczne SD | **1.44×** — czyli ±44 % na pojedynczej scenie |
   | zakres | 0.57× (`office_0`) – 1.65× (`hotel_0`) |

   Rozrzut nie jest artefaktem estymatora: dla każdej scenie wszystkie 80 par SS 1.0 dało skończone RT60.
   **Cytować to jako uzasadnienie porównania wewnętrznego** — jest mocniejsze niż argument z ostrożności.
   Jednocześnie oznacza, że SS 1.0 **nie może służyć jako kryterium poprawności** dla pojedynczej scenie:
   przy takim rozrzucie zarówno `frl_apartment_2` bez sufitu (0.49×), jak i załatana (1.36×) mieszczą się
   w rozrzucie scen o nienaruszonej geometrii. Szczegóły: `RAPORT_SESJI_2026-07-26_29.md` §2.14.
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
   zaniżył `N`). Jeśli obcięte próbki się pojawią, można je uzupełnić bez powtarzania generacji: pętla
   weryfikacyjna z §3.4 już je oznacza, więc wystarczy dorenderować je z podniesionym `N_MAX`.

   **Sprostowanie kryterium (2026-07-29).** Powyższy warunek `clamped == "max"` jest **za wąski** i przegapia
   rzeczywiste przypadki. `clamped` opisuje wyłącznie clamp na etapie *planowania* (`n_raw` vs `N_MIN`/`N_MAX`),
   tymczasem `N_MAX` jest także twardym limitem **całkowitej** liczby renderów próbki, więc wiąże również
   w pętli dorenderowującej — przy `n_planned < 40`, ale `n_planned + n_rendered_extra` dobijającym do 40.
   Poprawne kryterium to:

   ```
   n_total >= N_MAX  AND  snr_final < TARGET_SNR
   ```

   `--verify` stosuje to kryterium i wypisuje wszystkie próbki z `n_total >= N_MAX` z nazwy, także te, które
   mimo limitu próg osiągnęły.

   **Pierwszy zaobserwowany przypadek** (`office_1`, regeneracja 2026-07-29, 576 próbek): trzy próbki dobiły
   do `N_MAX`, z czego **jedna nie osiągnęła progu**:

   | lokalizacja | kąt | `n_planned` | `+extra` | `n_total` | `snr_probe` | `snr_final` |
   |---|---|---|---|---|---|---|
   | 5 | 160° | 25 | +15 | 40 | 2.758 | 3.851 ✓ |
   | 33 | 290° | 31 | +9 | 40 | 2.660 | **3.461 ✗** |
   | 35 | 170° | 21 | +19 | 40 | 2.555 | 4.750 ✓ |

   Udział: **1 na 576 = 0.17 %** w tej scenie. Żadna z tych lokalizacji nie miała `clamped == "max"`
   (`n_raw` = 25, 31, 21 — wszystkie poniżej 40), co potwierdza, że stare kryterium wykryłoby zero przypadków.
   Poprzednia wersja `office_1` (stara ścieżka) miała jedną próbkę przy `N_MAX`, która próg osiągnęła —
   różnica 1 vs 3 przy 576 próbkach nie jest rozróżnialna statystycznie.
7. **Kwantyzacja do `float16`** wnosi błąd 7–8·10⁻⁵ RMSE do każdej zapisanej próbki (§4.1). Jest 112–240×
   poniżej podłogi szumu, ale nie jest zerem — przy analizach porównujących różnice rzędu 10⁻⁴ trzeba o nim
   pamiętać. Osiągnięty SNR (`snr_probe`, `snr_final`) liczyć **przed** rzutowaniem.

8. **Sześć scen Replica nie ma sufitu — 46 % lokalizacji jest akustycznie otwartych**
   (zmierzone 2026-07-29, `measurements/rt60_vs_sabine.py`). Rodzina `frl_apartment_0..5` ma
   pokrycie sufitem **5–7 %** rzutu, pozostałe 12 scen 87–100 %. Skoro sufitu nie ma w siatce,
   nie ma go też w symulacji — energia ucieka górą i pogłos jest krótszy.

   | grupa | sceny | pokrycie sufitem | lokalizacji | mediana `sigma_1` |
   |---|---|---|---|---|
   | otwarte | `frl_apartment_0..5` | 5–7 % | **807 (46 %)** | 0.04570 |
   | zamknięte | pozostałe 12 | 87–100 % | 933 | 0.06322 |

   Rozdzielenie podłogi szumu jest **zupełne**: `max(otwarte) = 0.04892 < min(zamknięte) =
   0.05648`, Mann-Whitney **p = 0.00005**. Mechanizm: krótszy ogon pogłosowy → mniej odbić
   stochastycznych w oknie 60 ms → mniejsza wariancja Monte Carlo → niższe `N`.

   Konsekwencje: (a) echa z tych scen mają systematycznie inny charakter — to własność zbioru
   Replica, nie generatora; (b) jedna z trzech scen held-out (`frl_apartment_5`) jest otwarta,
   dwie zamknięte; (c) niższe `N` w tej rodzinie jest poprawną reakcją reguły adaptacyjnej,
   nie artefaktem; (d) wzory Sabine'a i Eyringa nie stosują się do tych scen.

   > **Uściślone przez ograniczenie 10.** Podział „6 otwartych / 12 zamkniętych" pochodzi
   > z pokrycia sufitem — heurystyki geometrycznej. Bezpośredni pomiar ucieczki promieni
   > pokazał, że grup jest **trzy**, a cztery ze scen nazwanych tu „zamkniętymi" nie są
   > akustycznie szczelne. Punkt (b) jest w tej formie mylący.

9. **Charakterystykę szumu zmierzono 0.21 m wyżej, niż idzie produkcja** (wykryte 2026-07-28, patrz §2).
   Wszystkie eksperymenty w `diagnose_rlr_noise.py` ustawiały pozycję agenta przez
   `pathfinder.snap_point()`, czyli na powierzchni navmesha, a produkcja stawia agenta na `y` z `graph.pkl`
   — o medianę 0.2125 m niżej. Dotyczy to `SIGNAL_10DEG = 0.0644`, rozkładu `N_raw` (mediana 9.83) i tempa
   0.2606 s/render. Jest to ten sam rodzaj zastrzeżenia co ograniczenie 5 (pomiary na 1.5 m, produkcja na
   1.25 m), z tą różnicą, że tu przesunięty jest **węzeł agenta**, a nie offset sensora — więc oba składają
   się na łączne ~0.46 m między historyczną a produkcyjną wysokością słuchacza nad podłogą.

   Przesłanki, że to nie unieważnia reguły adaptacyjnej: sygnał 10° okazał się niezależny od sceny
   **i** od orientacji (0.0639–0.0662 na 8 pozycjach w 4 scenach), a wielkością, która faktycznie się waha —
   szumem — reguła steruje adaptacyjnie, mierząc go **na miejscu**, przy produkcyjnej geometrii. Błąd
   w `SIGNAL_10DEG` przenosi się na `N` kwadratowo, ale weryfikacja po fakcie (§3.4) i tak koryguje próbki,
   które nie dobiją progu. Mimo to: jeśli `SIGNAL_10DEG` trafia do pracy jako liczba, należy go przemierzyć
   na produkcyjnej geometrii — jest to ~40 renderów na pozycję.

10. **Akustycznie szczelnych jest tylko 8 z 18 scen — 516 z 1740 lokalizacji (29.7 %)**
    (zmierzone 2026-07-29, `measurements/ray_escape_survey.py`). Ograniczenie 8 opierało się
    na pokryciu sufitem; ta liczba pochodzi z bezpośredniego pomiaru **ułamka promieni
    uciekających ze sceny** — wielkości, którą autorzy silnika wskazują jako właściwą miarę
    domknięcia (SoundSpaces 2.0, arXiv 2206.08312; API `RLRA_GetIndirectRayEfficiency()`
    istnieje w naszej kopii biblioteki, ale habitat-sim go nie eksponuje). Metoda zastępcza:
    równoprostokątny sensor głębi w punkcie słuchacza, 256 × 512 kierunków, udział kierunków
    bez trafienia ważony kątem bryłowym. Kontrola negatywna wbudowana — sceny szczelne dają
    ≤ 0.09 % w **każdej** lokalizacji.

    **Sceny szczelne — te można generować bez zastrzeżeń akustycznych:**

    | scena | lokalizacji | mediana ucieczki | max |
    |---|---|---|---|
    | `apartment_0` | 211 | 0.00 % | 0.01 % |
    | `office_4` *(held-out)* | 76 | 0.00 % | 0.01 % |
    | `room_0` | 57 | 0.00 % | 0.01 % |
    | `hotel_0` | 48 | 0.00 % | 0.01 % |
    | `room_2` | 47 | 0.00 % | 0.01 % |
    | `room_1` | 35 | 0.00 % | 0.00 % |
    | `office_0` | 26 | 0.02 % | 0.09 % |
    | `office_1` *(walidacyjna)* | 16 | 0.00 % | 0.01 % |
    | **razem** | **516** | | |

    **Pozostałe 10 scen — dwa różne mechanizmy przecieku:**

    | grupa | sceny | lokalizacji | mediana | max | lok. > 10 % | gdzie dziura |
    |---|---|---|---|---|---|---|
    | nieszczelne bokiem | `apartment_1`(176), `apartment_2`(142), `office_3`(62), `office_2`(37) | 417 (24.0 %) | 0.53–3.60 % | **48.3 %** | 0–36 % | horyzont i niżej |
    | bez sufitu | `frl_apartment_0..5` | 807 (46.4 %) | 21.8–23.6 % | 29.0 % | 89–93 % | **tylko** nad horyzontem |

    Rozkład kątowy rozdziela mechanizmy bezbłędnie: w `frl_apartment_*` pasmo > 60° elewacji
    ucieka w 99–100 %, a przy horyzoncie w 0.0 %; w czterech pozostałych scenach dokładnie
    odwrotnie (> 60°: 0.0 %; ucieczka w pasmach 10–30° i −10…10°, czyli przez przejścia, okna,
    niezeskanowane fragmenty ścian).

    **Najważniejsza korekta wobec ograniczenia 8:** `apartment_2` jest sceną **held-out**
    i ma 100 % pokrycia sufitem, ale **36 % jej lokalizacji traci ponad 10 % kąta bryłowego**
    (maksimum 41 %). Podobnie `apartment_1` (19 % lokalizacji > 10 %, maks. 48 %) i `office_2`
    (16 %). Jedynie `office_3` jest graniczne i praktycznie nieszkodliwe (maks. 4.5 %).
    `sigma_1` tego nie wykryło, bo przeciek boczny dotyka mniejszości lokalizacji w scenie
    i nie przesuwa mediany sceny — rozdzielenie `sigma_1` z ograniczenia 8 pozostaje prawdziwe,
    ale jest **grubsze**, niż zakładano.

    Skład podzbiorów: held-out ma dokładnie po jednej scenie każdego typu (`office_4` /
    `apartment_2` / `frl_apartment_5`), ale proporcje nie odpowiadają całości — sceny
    nieszczelne bokiem to 38.8 % held-outu wobec 24.0 % zbioru.

    Dane: `outputs/measurements/ray_escape/` (wersjonowane). Omówienie i konsekwencje dla
    pracy: `OBSERWACJE_METODOLOGICZNE.md` §1 („Uściślenie 2026-07-29").

    **Nieszczelność jest usuwalna — zmierzone, ale NIE stosowane w wariancie głównym.**
    Każda z 10 nieszczelnych scen ma 1–2 dziury o polu > 1 m² (reszta to brzegi mebli);
    `measurements/patch_scene_holes.py` domyka je wszystkie, sprowadzając ucieczkę promieni
    do 0.00–0.24 %. Na `frl_apartment_2` podnosi to RT60 z 0.48× do 0.96× wartości
    SoundSpaces 1.0 — ale przy **dopasowanym** materiale sufitu (semantycznie poprawny gips
    daje 1.48×), więc nie jest to walidacja. **Generacja produkcyjna idzie na geometrii
    ORYGINALNEJ**; łatanie jest opisane jako zmierzona opcja i ewentualny wariant dodatkowy.
    Szczegóły: `RAPORT_SESJI_2026-07-26_29.md` §2.13.

---

## 6. Otwarte, świadomie odłożone

1. **Config materiałów per scena** dla kategorii `floor` — podłogi drewniane w rodzinie apartamentów kontra
   szare w `office_0..4`, `room_0`, `room_2`. Kosztuje zero (i tak jeden Simulator na scenę), nieprzetestowane.
   `REPLICA_MATERIALS.md §6`.
2. **Walidacja wobec publicznych pomiarów rzeczywistych apartamentu FRL** (SoundSpaces 2.0, Sek. 5.2). To nasza
   rodzina scen; dałoby obiektywne kryterium doboru liczby promieni zamiast samoodniesienia do 5000.
3. E5 — mapa przestrzenna.

4. ~~**Symulacja audio wykonuje się DWA RAZY na render**~~ — **ZAMKNIĘTE 2026-07-28**, patrz §4.3.
   Zdublowane wywołanie usunięte po pomiarowym potwierdzeniu równoważności obu ścieżek; tempo spadło
   z 0.2606 do 0.1412 s/render, budżet z ~45 h do **≈ 31.9 h** (§4, liczba z pełnego census).

   Warta zapamiętania obserwacja poboczna z tamtego pomiaru: **RGB i depth są w tym potoku darmowe** —
   0.2 ms wobec 143 ms na samo audio, czyli 700×. Rozdzielanie obserwacji wizualnych od audio (renderowanie
   ich raz na orientację zamiast N razy) nie daje nic mierzalnego, mimo że rendering wizualny jest
   deterministyczny. Sprawdzono przy okazji, że takie rozdzielenie **nie zmieniłoby sekwencji RNG audio**:
   wywołanie `sensor.get_observation()` tylko dla sensora audio, z pominięciem `draw_observation()` kamer,
   daje IR identyczne co do bitu przez 5 kolejnych renderów w dwóch świeżo skonstruowanych instancjach
   `Simulator`. Nie ma to jednak zastosowania praktycznego.
