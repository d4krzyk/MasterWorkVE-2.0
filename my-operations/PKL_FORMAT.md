# Format `scene_observations_128.pkl` i odtworzony config kamery

Dokument opisuje strukturę pliku `scenes_ve_metadata_locations/scene_observations_128.pkl` oraz **dokładną
konfigurację sensorów, którą go wygenerowano** — odtworzoną wstecznie i zweryfikowaną testem spójności
piksel-po-pikselu. Config jest potrzebny, bo rozszerzenie datasetu z 4 do 36 orientacji wymaga dorenderowania
brakujących 32 orientacji **spójnie** z istniejącymi 4; gdyby RGB/depth różniły się między „starymi" a „nowymi"
orientacjami, model uczyłby się artefaktu renderowania zamiast zależności audio-wizualnej.

Ustalenia z 2026-07-25. Skrypty diagnostyczne były jednorazowe (nie wchodzą do repo); odtworzenie opisano
poniżej na tyle dokładnie, żeby dało się je powtórzyć.

## Pochodzenie pliku

Plik ma datę modyfikacji **2020-08-20** i **nie jest śledzony przez gita** — nie powstał w tym repozytorium.
To pobrany artefakt z oryginalnego wydania **VisualEchoes (Gao et al., ECCV 2020)**. Przeszukanie historii gita
oraz całego repo (poza katalogami vendored) nie ujawniło żadnego skryptu, który by go generował — jedyne
odwołanie to komórka wczytująca w `my-operations/check_data.ipynb`.

Najbliższym krewnym jest `sound-spaces/scripts/cache_observations.py` (ten sam ekosystem FAIR, ta sama siatka
punktów Replica), ale **nie jest to skrypt źródłowy**: zapisuje on obserwacje przepuszczone przez
`SensorSuite` habitat-laba, co daje RGB 3-kanałowe i depth przycięty/znormalizowany, podczas gdy nasz pkl ma
RGB **4-kanałowe** i depth **w surowych metrach bez klipowania**. Nasz plik zawiera więc surowy wynik
`sim.get_sensor_observations()`. Skrypt ten pozostaje jednak użyteczny jako świadectwo konwencji (klucz
`(node, angle)`, kąty `[0, 90, 180, 270]`, `quat_from_angle_axis` wokół osi Y) — i te konwencje potwierdziły
się empirycznie.

## Struktura pliku

```
observations: defaultdict
  └─ [scene_name: str]                  # 18 scen Replica
       └─ [(location_id: int, orientation_deg: int)]
            └─ {"rgb": ndarray, "depth": ndarray}
```

- **18 scen**: `apartment_0..2`, `frl_apartment_0..5`, `hotel_0`, `office_0..4`, `room_0..2`
- **1740 lokalizacji** łącznie (suma po scenach), **6960 kluczy** = 1740 × 4 orientacje
- **orientacja**: `int`, w **stopniach**, ze zbioru `{0, 90, 180, 270}` (nie indeks)

### `rgb`
- kształt `(128, 128, 4)`, dtype `uint8`, zakres 0–255
- **4 kanały: RGBA** w kolejności R, G, B, A. Kanał alfa jest stały = 255 (potwierdzone: min = max = 255).
  Do użytku modelowego bierzemy `rgb[..., :3]`.

### `depth`
- kształt `(128, 128)` (bez wymiaru kanału), dtype `float32`
- **jednostki: metry**, wartości surowe — **nieznormalizowane i nieklipowane**
- zaobserwowane globalne ekstrema na próbce ze wszystkich 18 scen: **min 0.0, max 12.66 m**
- Max 12.66 m > typowego `MAX_DEPTH = 10.0` habitat-laba dowodzi, że **nie zastosowano klipowania** — to
  kolejny dowód, że pkl omija `SensorSuite`. Generator musi produkować depth w tej samej surowej skali
  metrowej; **nie normalizować i nie klipować**.
- Wartości 0.0 występują (piksele bez trafienia w geometrię).

### Klucz `location_id` — mapowanie na `points.txt`
`location_id` to **wartość kolumny `id` z `points.txt`**, nie indeks wiersza i nie indeks porządkowy.
Zweryfikowane: dla każdej sceny zbiór `location_id` z pkl zawiera się w zbiorze `id` z `points.txt`.

Konwersja współrzędnych jest ta sama co zawsze w tym projekcie (patrz CLAUDE.md): `x = a`, `z = -b`.
Sprawdzone na **wszystkich 18 scenach i wszystkich węzłach grafu naraz**: maksymalna rozbieżność między
`graph.pkl`'s `point[0] / point[2]` a `points.txt`'s `(a, -b)` wynosi **dokładnie 0.0** dla obu osi w każdej
scenie — współrzędne są identyczne co do bitu. `points.txt` jest więc w pełni równoważnym (i szerszym) źródłem
pozycji; jedyne, czego nie zawiera, to wysokość `y`.

**`graph.pkl` jest przefiltrowanym podzbiorem i nie wystarcza**: w `room_0` pkl ma 57 lokalizacji, a graf
tylko 51 (brakujące id: **102, 103, 111, 112, 120, 121**); w `room_1` 35 vs 33 (brakujące: **45, 51**).
W pozostałych 16 scenach zbiory pokrywają się dokładnie. Łącznie **8 z 1740** lokalizacji istnieje w pkl, ale
nie w grafie — generator musi je obsłużyć, więc źródłem pozycji jest `points.txt`, nie `graph.pkl`.

### Wysokość agenta (`y`)
`points.txt` nie przechowuje wysokości. Okazuje się, że **`y` jest stałe w obrębie sceny** — sprawdzone we
wszystkich 18 scenach, w każdej `graph.pkl` ma dokładnie jedną unikalną wartość `y`. Dlatego dla lokalizacji
spoza grafu wystarczy wziąć tę samą stałą sceny (nie trzeba `pathfinder.snap_point`, który mógłby dać
minimalnie inną wartość i rozspójnić dane).

| scena | y | scena | y | scena | y |
|---|---|---|---|---|---|
| apartment_0 | -1.543482 | frl_apartment_3 | -1.538082 | office_1 | -1.016517 |
| apartment_1 | -1.725024 | frl_apartment_4 | -1.548102 | office_2 | -1.267472 |
| apartment_2 | -1.659589 | frl_apartment_5 | -1.470114 | office_3 | -1.249195 |
| frl_apartment_0 | -1.642777 | hotel_0 | -1.075138 | office_4 | -1.238161 |
| frl_apartment_1 | -1.527613 | office_0 | -0.897211 | room_0 | -1.549177 |
| frl_apartment_2 | -1.514732 | room_1 | -1.441063 | room_2 | -2.486792 |

## Odtworzony config renderowania

```python
import quaternion            # MUSI być przed habitat_sim (CLAUDE.md)
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

cfg = habitat_sim.SimulatorConfiguration()
# UWAGA: mesh_semantic.ply, NIE replica_stage.stage_config.json - patrz niżej
cfg.scene_id = f"sound-spaces/data/scene_datasets/replica/{scene}/habitat/mesh_semantic.ply"
cfg.create_renderer = True
cfg.enable_physics  = False
cfg.gpu_device_id   = 0

specs = []
for uuid, stype in (("rgb",   habitat_sim.SensorType.COLOR),
                    ("depth", habitat_sim.SensorType.DEPTH)):
    s = habitat_sim.CameraSensorSpec()
    s.uuid        = uuid
    s.sensor_type = stype
    s.resolution  = [128, 128]
    s.position    = [0.0, 1.25, 0.0]   # wysokość kamery - NIE domyślne 1.5
    s.hfov        = 90.0
    specs.append(s)

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = specs

# ustawienie pozy agenta:
st = agent.get_state()
st.position = [x, y_sceny, z]                                             # x=a, z=-b z points.txt
st.rotation = quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))
st.sensor_states = {}                    # wyczyszczenie, żeby offset sensora policzył się od nowa
agent.set_state(st, True)
obs = sim.get_sensor_observations()      # surowe - bez SensorSuite
```

Dwa parametry, które **odbiegają od tego, co wydawało się oczywiste**, i które trzeba było ustalić empirycznie:

1. **Wysokość kamery 1.25 m, nie 1.5 m.** Domyślna wartość w `habitat_sim.CameraSensorSpec` to `[0, 1.5, 0]`;
   pkl używa jednak `1.25`, czyli domyślnej wartości **habitat-laba** (`SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]`
   w `habitat-lab/habitat/config/default.py:664`). To był jedyny realny błąd w pierwszej próbie odtworzenia.
2. **Źródło sceny to `mesh_semantic.ply`, nie `replica_stage.stage_config.json`.** Replica ma dwie ścieżki
   renderowania o wyraźnie różnym wyglądzie: teksturowany PTex (przez stage config) i siatkę z kolorami
   wierzchołków (`mesh_semantic.ply`). VisualEchoes/SoundSpaces 1.0 używały tej drugiej.

Rozdzielczość 128×128 i hfov 90° są zbieżne z domyślnymi wartościami `CameraSensorSpec`.

## Test spójności (weryfikacja odtworzenia)

Metoda: renderowanie lokalizacji i orientacji, **które już istnieją w pkl**, odtworzonym configiem, a następnie
porównanie piksel-po-pikselu. Rendering wizualny jest deterministyczny dla danego configu (inaczej niż audio
RLR, gdzie szum Monte Carlo jest realny), więc każda niezerowa różnica oznaczałaby rozbieżność configu.

**Wynik na `room_0` (20 par lokalizacja × kąt):**

| metryka | wartość |
|---|---|
| RGB RMSE (skala 0–255) | **0.0077** średnio, 0.0146 max |
| pikseli RGB identycznych co do bitu | **99.992 %** (min 99.979 %) |
| depth RMSE | **2.4 × 10⁻⁵ m** średnio, 6.2 × 10⁻⁵ m max |

Resztkowa różnica to pojedyncze piksele różniące się o 1 LSB oraz szum zaokrąglenia `float32` w depth —
poniżej progu jakiegokolwiek znaczenia dla uczenia modelu.

**Kontrola negatywna** (ten sam test, ale hfov = 70 zamiast 90): RGB RMSE **33.59**, depth RMSE **0.478 m**,
tylko 28.8 % pikseli identycznych. Dopasowanie przy hfov = 90 nie jest więc przypadkiem — parametry są
rzeczywiście zidentyfikowane, a nie „dowolne, byle blisko".

**Uogólnienie na inne sceny** (po 12 par każda, lokalizacje z grafu):

| scena | RGB RMSE (śr / max) | depth RMSE (śr / max) |
|---|---|---|
| room_0 | 0.0079 / 0.0146 | 2.5 × 10⁻⁵ / 6.2 × 10⁻⁵ m |
| office_0 | 0.0100 / 0.0166 | 1.6 × 10⁻⁵ / 4.4 × 10⁻⁵ m |
| frl_apartment_0 | 0.0097 / 0.0179 | 3.6 × 10⁻⁵ / 1.1 × 10⁻⁴ m |

**Lokalizacje spoza `graph.pkl`** — wszystkie 8, pozycja wyłącznie z `points.txt` + stała wysokość sceny
(maksimum po 4 kątach na lokalizację):

| scena | location_id | RGB RMSE max | depth RMSE max |
|---|---|---|---|
| room_0 | 102 | 0.1746 | 5.3 × 10⁻³ m |
| room_0 | 103, 111, 112, 120, 121 | 0.0087 – 0.0199 | 6.6 – 7.0 × 10⁻⁵ m |
| room_1 | 45, 51 | 0.0124 – 0.0156 | 2.6 – 3.2 × 10⁻⁵ m |

Siedem z ośmiu jest nieodróżnialnych od lokalizacji z grafu (odniesienie w tych samych scenach: RGB
0.0087–0.0141, depth 4.8–5.9 × 10⁻⁵ m). Odstaje wyłącznie `room_0/loc=102` — i **nie jest to problem pozycji**,
patrz niżej.

### Dlaczego `room_0/loc=102` odstaje (i dlaczego to nieistotne)

Punkt ten wymagał osobnego wyjaśnienia, bo przy jednym uruchomieniu dawał depth RMSE 5.3 × 10⁻³ m, a przy innym
6.4 × 10⁻⁵ m — pozorna niepowtarzalność, która podważałaby cały test. Przyczyna okazała się inna, niż sugeruje
intuicja:

- **Rendering jest w pełni deterministyczny.** Pięć kolejnych renderów tej samej pozy w jednej sesji oraz
  rendery z drugiej, świeżo skonstruowanej sesji dały wyniki **identyczne co do bitu**
  (`max |render_i − render_0| = 0.0`). Kolejność renderowania również nie ma wpływu (ten sam wynik niezależnie
  od tego, jaki render go poprzedzał). Nie ma tu żadnej losowości — w przeciwieństwie do audio RLR.
- **Cała różnica brała się z 3 × 10⁻⁷ m różnicy w `y`**: `-1.5491767` (wartość `float32` z `graph.pkl`) vs
  `-1.549177` (ta sama liczba zaokrąglona do 6 miejsc). Przy `loc=102, kąt=90°` kamera patrzy na powierzchnię
  pod skrajnie małym kątem (sylwetka / krawędź na granicy widoczności), więc przesunięcie o ułamek mikrometra
  przełącza **1–2 piksele** między dwiema bardzo różnymi głębokościami. Frakcja pikseli różniących się o więcej
  niż 1 mm wynosi 0.0001, czyli ok. 1.6 piksela na 16384.
- RMSE jest tu myląca jako miara: pojedynczy piksel różniący się o ~2 m podnosi RMSE do 5 mm, choć obraz jest
  praktycznie identyczny. Dla porządku: to zjawisko dotyczy pikseli na krawędziach sylwetek i wystąpi zawsze,
  niezależnie od configu.

**Wniosek praktyczny dla generatora:** brać `y` z `graph.pkl` w pełnej precyzji `float32` tam, gdzie węzeł
istnieje, i tę samą stałą sceny dla 8 lokalizacji spoza grafu. Wybór między tymi dwiema wartościami `y` jest
poniżej rozdzielczości, jaką da się wstecznie ustalić z pkl, i wpływa najwyżej na pojedyncze piksele na
krawędziach.

**Konwencja kąta zweryfikowana osobno**: dla każdej testowanej lokalizacji porównano *każdy* renderowany kąt
z *każdym* kątem z pkl. Macierz błędów ma minimum wyłącznie na przekątnej (0.0015–0.6 m) przy wartościach
poza przekątną 1.0–2.5 m, we wszystkich testowanych lokalizacjach. Mapowanie jest więc identycznościowe:
`orientacja z pkl` = argument `quat_from_angle_axis(deg2rad(·), [0,1,0])`, w stopniach.

## Werdykt

**BEZPIECZNY.** Config jest odtworzony z dokładnością do szumu zaokrąglenia zmiennoprzecinkowego i uogólnia się
na inne sceny oraz na lokalizacje spoza grafu. Brakujące 32 orientacje można dorenderować spójnie z
istniejącymi 4.

## Konsekwencja dla sensora audio: wysokość 1.25 m

Odtworzona wysokość kamery (1.25 m) wymusza decyzję po stronie akustyki. `test_rlr_audio.build_simulator()`
ustawiał dotąd `AudioSensorSpec.position = [0, 1.5, 0]` — wartość wziętą z konwencji SoundSpaces (por.
`sound-spaces/soundspaces/continuous_simulator.py:341`). Powstawał więc **25 cm rozjazdu między punktem
obserwacji wizualnej i akustycznej**, będący przypadkowym zbiegiem dwóch różnych wartości domyślnych.

Zmierzone (room_0, 3 pozycje, N=10, materiały włączone — eksperyment `--exp listener_height`):

| wielkość | RMSE spektrogramu | po odszumieniu |
|---|---|---|
| efekt przesunięcia słuchacza 1.25 ↔ 1.5 m | 0.0697 | 0.0659 |
| sygnał obrotu o 10° | 0.0684 | 0.0646 |
| szum resztkowy przy N=10 | 0.0225 | — |

Przesunięcie o 25 cm zmienia echo **1.02× tak mocno jak pełny obrót o 10°** — czyli tak samo mocno jak efekt,
który cała praca próbuje rozdzielić, i 3.1× ponad poziom szumu. To nie jest szczegół kosmetyczny.

**Decyzja: słuchacz audio na 1.25 m, zrównany z kamerą** (`AudioSensorSpec.position = [0, 1.25, 0]`). Uzasadnienie:
1.25 m jest twardo narzucone (tylko ta wartość odtwarza pkl co do bitu), podczas gdy 1.5 m to wyłącznie
konwencja; a agent ucieleśniony powinien widzieć i słyszeć z jednego punktu.

**Zastrzeżenie:** wszystkie dotychczasowe pomiary szumu (E1–E4, checkpoint-boundary) wykonano przy słuchaczu na
1.5 m. Statystyka szumu Monte Carlo nie powinna zależeć od 25 cm, ale nie zostało to przemierzone na 1.25 m —
jeśli któraś liczba trafia do pracy jako dokładna, należy ją najpierw powtórzyć na wysokości produkcyjnej.
