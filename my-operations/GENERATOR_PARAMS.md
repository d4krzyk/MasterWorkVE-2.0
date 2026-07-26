# Parametry generatora datasetu — FINALNE

Specyfikacja zamrożona **2026-07-26**, po zamknięciu fazy charakteryzacji i walidacji.
To jest dokument referencyjny dla sesji, w której powstanie generator — **nie jest to kod generatora**.

Każda pozycja ma odwołanie do eksperymentu, który ją rozstrzygnął. Surowe wyniki:
`my-operations/Replica/diagnose_rlr_noise_out/diagnostics_report.json` (klucz podany przy każdej pozycji).
Konteksty: `PKL_FORMAT.md` (kamera i zbiór lokalizacji), `REPLICA_MATERIALS.md` (materiały akustyczne),
`CLAUDE.md` (środowisko, pułapki GPU).

---

## 1. Tabela parametrów

| parametr | wartość | źródło |
|---|---|---|
| `indirect_ray_count` | **500** | `e2_bias_orientation`, `e2_rays_vs_renders` |
| `thread_count` | **1** | `e2_thread_budget_confirm` |
| `n_renders` | **adaptacyjne per lokalizacja**, patrz §3 | `noise_floor_scenes`, `noise_floor_orientation` |
| `averaging_domain` | **`"mag"`** = (1/N)Σ\|STFT\| | `e3_averaging_domain` |
| `material_config` | **`my-operations/replica_material_config.json`** | `materials_verify` |
| `listener_height` | **1.25 m** (kamera i audio) | `listener_height`, `PKL_FORMAT.md` |
| `simulator_rotation` | **nie dotyczy** — brak wycieku | `gpu_memory_scale` |
| `est_time_total` | **≈ 44 h** (+ dorenderowanie, §3.4) | §4 |
| `est_disk` | **≈ 30 GB** | §4 |

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
spektrogram: (2, 257, 166) float32
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
jest trudno. Rozkład wymaganego N po 12 zmierzonych pozycjach: **4, 6, 7, 7, 9, 9, 9, 10, 11, 12, 12, 21**.

| schemat | czas | pokrycie |
|---|---|---|
| **adaptacyjne** | **44.2 h** | **100 %** |
| stałe N=12 | 54.4 h | 92 % |
| stałe N=18 | 81.6 h | 92 % — zdominowane, w rozkładzie nie ma nic między 12 a 21 |
| stałe N=21 | 95.2 h | 100 % |

Adaptacyjne jest **2.15× tańsze niż stałe N o tej samej gwarancji**. Jest też odporne na to, że rozkład znamy
z 12 z 1740 lokalizacji — samo dostosuje się do pozycji, których nie zmierzyliśmy. Stałe N nie jest.

### 3.2 Kryterium

Wszystko liczone z renderów, które i tak powstają — **żaden dodatkowy pomiar wstępny nie jest potrzebny**.
Estymator szumu to ten sam podział na dwie rozłączne połówki, którego używa każdy eksperyment w tym projekcie:

```
sigma_1 = RMSE(polowka_A, polowka_B) / sqrt(2) * sqrt(n/2)     # szum POJEDYNCZEGO renderu
N       = clamp( ceil( (TARGET_SNR * sigma_1 / SIGNAL_10DEG)^2 ), N_MIN, N_MAX )

TARGET_SNR   = 3.5
SIGNAL_10DEG = 0.0644      # mediana z noise_floor_scenes, stała między scenami
N_MIN, N_MAX = 6, 24
```

`RMSE(A, B) = sqrt(2)·sigma_N` — stąd dzielenie przez `sqrt(2)`. Nigdy nie porównywać surowych RMSE dwóch
zaszumionych estymat bez tej dekompozycji (błąd popełniony raz w Bloku B).

### 3.3 Procedura per lokalizacja

1. **Sonda**: 8 renderów przy orientacji 0°, podział 4+4 → `sigma_1` → `N`.
2. Renderuj `N` dla **wszystkich 36 orientacji** tej lokalizacji, **wykorzystując rendery sondy** dla
   orientacji 0°. Odzyskiwane w całości gdy `N ≥ 8`; strata gdy `N < 8` to w sumie **0.06 h** na cały zbiór.
3. Estymata próbki = `mean(|STFT|)` po `N` renderach (domena `mag`).

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

### 3.5 Sformułowanie do pracy

> Sygnał różnicujący orientacje jest niezależny od sceny (RMSE spektrogramu 0.0639–0.0662 na ośmiu pozycjach w
> czterech scenach testowych), natomiast podłoga szumu Monte Carlo waha się 2.7-krotnie i jest własnością
> pozycji nasłuchu, a nie orientacji (rozrzut wewnątrz pozycji 5.5 % wobec 84 % między pozycjami). Stała liczba
> renderów musi więc albo marnować obliczenia tam, gdzie jest łatwo, albo nie osiągać założonego SNR tam, gdzie
> jest trudno. Zastosowano zamiast tego adaptacyjne próbkowanie sterowane wariancją — technikę standardową w
> renderingu Monte Carlo — z progiem wyznaczanym per lokalizacja z podziału renderów na dwie niezależne
> połówki, co daje jednorodne SNR ≥ 3.5 przy 2.15-krotnie mniejszym koszcie niż stała liczba renderów o tej
> samej gwarancji.

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
czas  = 62 640 probek x 9.75 renderu (srednia) x 0.2606 s  ~ 44.2 h   + dorenderowanie z 3.4
dysk  = 21.4 GB spektrogramow float32 (2,257,166)
      +  8.2 GB RGB uint8 + depth float32 (128x128)
      = 29.6 GB                                  (wolne na /home: 282 GB)
```

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

---

## 6. Otwarte, świadomie odłożone

1. **Config materiałów per scena** dla kategorii `floor` — podłogi drewniane w rodzinie apartamentów kontra
   szare w `office_0..4`, `room_0`, `room_2`. Kosztuje zero (i tak jeden Simulator na scenę), nieprzetestowane.
   `REPLICA_MATERIALS.md §6`.
2. **Walidacja wobec publicznych pomiarów rzeczywistych apartamentu FRL** (SoundSpaces 2.0, Sek. 5.2). To nasza
   rodzina scen; dałoby obiektywne kryterium doboru liczby promieni zamiast samoodniesienia do 5000.
3. E5 — mapa przestrzenna.
