# Odtworzenie zbioru (karta Hugging Face) — ustalenia

Zbiór: <https://huggingface.co/datasets/d4krzyk/echo-angular-density-replica> (dostęp warunkowy — Replica Research Terms, tylko użycie niekomercyjne).

Sporządzone 2026-11-19, **tylko odczyt**: odczytano atrybuty HDF5, `git log`/`git diff`, czasy modyfikacji
plików i dokumentację. Generator nie był uruchamiany. Każde ustalenie ma znacznik źródła:
**[atrybut]** — zapisane w pliku HDF5, **[log]** — `queue.log`/`generate.log`/`git`, **[mtime]** — czas
modyfikacji pliku na dysku, **[dok]** — dokumentacja (nie weryfikowana tu pomiarem), **[brak dowodu]**.

---

## 1. Commit, z którego wygenerowano dane

### 1.1 Atrybuty HDF5

Pełna lista atrybutów (`office_1.h5` z `main`, `office_2.h5` z `patched`) nie zawiera żadnego pola z hashem
commita repo pracy (brak `git_commit`, `code_version` itp.). Jedyne pola o kodzie to:

| atrybut | `main` / `office_1` | `patched` / `office_2` |
|---|---|---|
| `script` / `script_version` | `generate_echo_dataset.py` / `1.0.0` | to samo |
| `habitat_sim_commit` | `80f8e31140eaf50fe6c5ab488525ae1bdf250bd9` (0.2.2) | to samo |
| `rlr_audio_propagation_commit` | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | to samo |
| `variant` | `main` | `patched` |
| `scene_id` | `sound-spaces/data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply` | `outputs/patched_scenes/office_2/habitat/mesh_semantic.ply` |
| `created_utc` | 2026-07-29T23:12:20Z | 2026-07-31T13:18:23Z |
| `last_update_utc` | 2026-07-29T23:42:38Z | 2026-07-31T14:14:39Z |
| `chirp_sha256` | `d89558c1…0efca` | to samo |
| `material_config_sha256` | `49f90b4f…c532…` (`…974c132`) | to samo |
| `numpy_version` / `python_version` / `platform` | 1.26.4 / 3.10.14 / `Linux-6.8.0-136-generic-x86_64-with-glibc2.35` | to samo |

Wszystkie 28 plików: ten sam `script_version`, te same commity habitat-sim i RLR, `runs` ma dokładnie
**jeden** wpis (jedno uruchomienie, bez wznowień), `warmup_discard = 500`. Czyli wszystkie pliki powstały
w jednym trybie pracy kodu. Zakresy czasu: `main` 2026-07-29 23:12 → 2026-07-31 06:46 UTC,
`patched` 2026-07-31 07:09 → 2026-08-01 08:12 UTC.

### 1.2 Ustalenie z historii git

Czasy z `git log` są lokalne (CEST, +0200); atrybuty HDF5 są w UTC (CEST = UTC+2).

| commit | data | zawartość istotna dla generatora |
|---|---|---|
| `d80dd4bc` | 2026-07-28 18:54 | pierwszy generator + `echo_ctl.py` |
| `b01cd0b8` | 2026-07-29 00:17 | rozgrzewka symulatora (`WARMUP_DISCARD`) |
| `0c2b6d91` | 2026-07-29 15:37 (13:37 UTC) | restrukturyzacja; `params.py` z `WARMUP_DISCARD=500`, `N_MAX=64` |
| *(start `main`)* | 2026-07-30 01:12 lokalnie (= 07-29 23:12 UTC) | |
| *(koniec `main`)* | 2026-07-31 08:46 lokalnie | |
| *(start `patched`)* | 2026-07-31 09:09 lokalnie | |
| *(koniec `patched`)* | 2026-08-01 10:12 lokalnie | |
| `ccf211c4` | **2026-08-01 17:26** | wariant `main`/`patched`, `patch_scene_holes.py` |

**Wniosek: żaden commit nie istniał w chwili generacji.** Oba warianty powstały z drzewa roboczego, które
było `0c2b6d91` + zmiany jeszcze niezacommitowane, a te zmiany weszły dopiero w `ccf211c4`. Dowody:

1. `0c2b6d91` **nie zawiera** obsługi wariantu (`grep variant` w `generate_echo_dataset.py` — brak),
   a plik `main` ma atrybut `variant = main` i `queue.log` mówi „wariant main" **[log, atrybut]**.
2. Czasy modyfikacji plików z obsługą wariantu: `paths.py` 07-30 00:52, `scenes.py` 00:52, `renderer.py`
   00:53, `status.py`/`store.py` 00:53 (lokalnie) — **przed** pierwszym plikiem `main` (01:12) **[mtime]**.
   Siatki `patched` powstały 07-30 00:40, a `patch_scene_holes.py` ma mtime 00:39 **[mtime]**.
3. `generate_echo_dataset.py` i `echo_core/audio.py` mają mtime 2026-08-01 17:11 — **po** generacji obu
   wariantów, na 15 minut przed `ccf211c4`. Nie da się już ustalić, jak wyglądały te dwa pliki w czasie
   generacji; wiadomo tylko, że `--variant` w `generate_echo_dataset.py` musiał już istnieć (inaczej
   `echo_ctl.py`, który go przekazuje, kończyłby się błędem argparse) **[brak dowodu na resztę]**.

**Najlepsze przybliżenie, wspólne dla `main` i `patched`:**

- krótki: `ccf211c4`
- pełny: `ccf211c41eec93ee62278497ddbfcb46854fc372` (rodzic: `0c2b6d91c1d7f1d7ac2ad99bff73f06826a4f88e`)

To **nie jest** commit, z którego fizycznie uruchomiono generator (on jeszcze nie istniał). To pierwszy commit,
który zawiera kod generatora w stanie równoważnym wynikowo. Uzasadnienie równoważności poniżej.

### 1.3 Czy kod zmieniał się od zamrożenia parametrów (2026-07-26) do końca generacji?

Tak. `GENERATOR_PARAMS.md` zamrożono 07-26, ale generator powstał dopiero 07-28, więc całe jego życie
przypada po zamrożeniu. Zmiany po 07-26, które **wpływają na wynik**, weszły przed startem `main`
i są już w `0c2b6d91`:

- usunięcie zdublowanej symulacji audio (`audio_sims_per_render = 1`) — `GENERATOR_PARAMS.md` §4.3;
- `WARMUP_DISCARD = 500` (pierwszy przebieg `office_1` z 07-28 miał rozgrzewkę 20 renderów i był
  **nadpisany** `--force` przebiegiem z 07-30 01:12; `generate.log` zawiera oba);
- `N_MAX = 64` (24 → 40 → 64), `N_MIN = 6`, `N_PROBE = 8`, `TARGET_SNR = 3.5`, `SIGNAL_10DEG = 0.0644`;
- `y` z `graph.pkl`, nie z `snap_point()` (§2, poprawka 2026-07-28).

Zmiany między `0c2b6d91` a `ccf211c4` (`git diff`, pliki generatora i `echo_core/`): wyłącznie
**obsługa wariantu** (`--variant`, `set_variant`, `scenes_for_variant`, `paths.OUT_ROOT` przez moduł,
atrybut `variant`, wybór siatki w `scene_mesh()`), `dry_run` (`statvfs`) oraz zmiany komentarzy/docstringów
(usunięte odwołania do `CLAUDE.md`). `params.py`, `noise.py`, `spectrogram.py`, `runtime.py`, `verify.py`
**nie zmieniły się** (nie występują w diffie). Dla `--variant main` `scene_mesh()` zwraca tę samą ścieżkę
co przed zmianą. **Zmiany nie wpływają na wynik numeryczny** **[log: git diff]**.

Pozostałe wejścia (inne niż kod): `local_changes.patch` i skompilowane rozszerzenie
(`habitat_sim_bindings*.so`) mają mtime 2026-07-20, czyli sprzed generacji; submoduł RLR jest na `4fd446b`
= wartość z atrybutu **[mtime, git]**. Uwaga: `habitat-sim` ma niezacommitowane lokalne zmiany
(`git -C habitat-sim status`) — ich zapis to `habitat-sim/local_changes.patch`, śledzony w repo pracy
w commicie `ae27cf09`; `habitat_sim_commit` w atrybucie (`80f8e311`) opisuje więc bazę, a nie pełny stan.

**Różnica `main` vs `patched`:** nie wykryto — ten sam kod, ten sam commit. Różni je wyłącznie siatka
(`scene_id`), zgodnie z `GENERATOR_PARAMS.md` §4.5.

---

## 2. Komenda uruchomienia generatora

**Ustalone [log, kod]:** zbiór generowano przez `echo_ctl.py queue` (nadzorca w tle), który dla każdej
sceny wywołuje `generate_echo_dataset.py --scene <s> --variant <v>` (z `--resume`, jeśli plik już istnieje),
po jednym procesie OS na scenę. `queue.log` zaczyna się od „KOLEJKA start, wariant main, sceny: …".
Nie ma skryptu `.sh` ani zapisu dokładnej linii poleceń, którą wpisano — dokładna kolejność scen w
kolejce wynika z `queue.log`; nie jest ona istotna dla treści plików (każda scena to osobny proces
ze świeżym `Simulator`).

**Uwaga o ścieżkach.** Kod **nie czyta `$REPLICA_DIR` ani `$OUT_DIR`**: ścieżki są wpisane względem korzenia
repo (`echo_core/paths.py`). Zmienne poniżej to konwencja karty; trzeba dowiązać dane pod te ścieżki:

```
Replica     → <repo>/sound-spaces/data/scene_datasets/replica/<scena>/habitat/mesh_semantic.ply
lokalizacje → <repo>/scenes_ve_metadata_locations/scene_observations_128.pkl   (sha256 d944ef0c…6671; plik z VisualEchoes, NIE jest w repo)
punkty      → <repo>/my-operations/metadata/replica/<scena>/{points.txt,graph.pkl}   (w repo)
wyjście     → <repo>/outputs/echoes_36deg/<scena>/<scena>.h5            (main)
              <repo>/outputs/echoes_36deg_patched/<scena>/<scena>.h5    (patched)
siatki patched → <repo>/outputs/patched_scenes/<scena>/habitat/mesh_semantic.ply
```

### Minimalne komendy

```bash
conda activate habitat
cd $REPO        # korzeń repo pracy; PYTHONPATH ustawia sam echo_core/paths.py

# --- main: jedna scena ---
python my-operations/generate_echo_dataset.py --scene office_1 --variant main
# --- main: wszystkie 18 scen, jedna po drugiej (po jednym procesie na scenę) ---
python my-operations/echo_ctl.py --variant main queue all    # zakłada kolejkę i startuje nadzorcę w tle

# --- patched: najpierw zalatać siatki (raz), potem generować ---
python my-operations/measurements/patch_scene_holes.py --all
python my-operations/generate_echo_dataset.py --scene office_2 --variant patched
python my-operations/echo_ctl.py --variant patched queue all   # 10 scen z łatą
```

Poniżej wersja bez nadzorcy — pętla po scenach z `generate_echo_dataset.py`:

```bash
for s in apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 \
         frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 office_1 office_2 \
         office_3 office_4 room_0 room_1 room_2; do
  python my-operations/generate_echo_dataset.py --scene "$s" --variant main || break
done
# patched: tylko apartment_1 apartment_2 frl_apartment_0..5 office_2 office_3, z --variant patched
```

Pomocnicze (bez GPU): `--status`, `--verify <scena>`, `--dry-run --scene <s>`. Siatki `patched` zapisano
poleceniem z klasą łaty wg reguły domyślnej; dokładna linia (`--all`?) nie jest zapisana poza
`patch_report.json` (`min_area_m2 = 1.0`) — **[brak dowodu]**, wartości domyślne skryptu są zgodne z tym raportem.

### Środowisko

- conda `habitat`, `configs/environment-final.yml`: Python **3.10.14**, numpy **1.26.4**, scipy **1.15.3**,
  h5py **3.14.0** (HDF5 1.14.6), librosa 0.11.0, soundfile 0.14.0, opencv-python 4.10.0.84, numpy-quaternion
  2024.0.13. (W atrybutach zapisano tylko python/numpy/platform; reszta z pliku środowiska — **[dok]**.)
- **habitat-sim 0.2.2, baza `80f8e31140eaf50fe6c5ab488525ae1bdf250bd9`** + `habitat-sim/local_changes.patch`
  (m.in. `UploadMesh`, null-check `category()`, `import quaternion` przed rozszerzeniem).
- **rlr-audio-propagation `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`**.
- Budowa: `TARGET_HEADLESS=OFF`, `BUILD_WITH_BULLET=ON`, `BUILD_WITH_CUDA=ON`, `BUILD_WITH_AUDIO=ON`,
  `CMAKE_CUDA_ARCHITECTURES=90` (CLAUDE.md).
- Sprzęt: jedna GPU wymagana przez EGL (RTX 5070 Ti, sterownik 580.x; sterownik z chwili generacji nie
  jest zapisany w plikach — dziś 580.178.04). Linux 6.8.0-136-generic, glibc 2.35.

---

## 3. Czy SHA256 pliku jest powtarzalny

### 3.1 Pola niedeterministyczne w pliku (zmierzone w atrybutach i typach)

| miejsce | pole | deterministyczne? |
|---|---|---|
| atrybuty pliku | `created_utc`, `last_update_utc` | **nie** (zegar) |
| | `runs` (JSON z `started_utc`, `ended_utc`, `wall_seconds`, `hostname`, `exit_code`) | **nie** |
| | `hostname`, `platform` | **nie** (zależą od maszyny) |
| | `seconds_per_render`, `render_seconds_total` | **nie** (czas ścienny) |
| | `renders_total` | zależy od danych (liczba renderów) |
| `locations/` | `seconds`, `probe_seconds` | **nie** (czasy ścienne) |
| atrybuty | `chirp`, `material_config`, `scene_id` | **ścieżki względne do repo** — stałe, o ile układ repo ten sam; `scene_id` różni się między wariantami z założenia |
| obiekty HDF5 | `echo`, `rgb`, `depth`, `position`: `get_obj_track_times() == 1` | **nie** — HDF5 zapisuje czas modyfikacji obiektu w nagłówku |
| sąsiednie pliki | `decisions.jsonl`, `generate.log`, `progress.json`, `stdout.txt` | zawierają czasy; poza paczką |

Czyli **bajty pliku `.h5` różnią się przy każdym uruchomieniu niezależnie od zawartości**: zegar w atrybutach
i znaczniki czasu obiektów w nagłówkach HDF5 (dla `echo`, `rgb`, `depth`, `position` ustawione `track_times`).
To jest stwierdzone bezpośrednio na plikach.

### 3.2 Czy datasety `echo`/`rgb`/`depth` są bit-identyczne przy ponownej generacji

**Nie zweryfikowano pełną regeneracją** — nie uruchamiano generatora i nie ma w repo wyniku takiego testu
na całej scenie. Przesłanki:

- **za:** `threadCount = 1` jest warunkiem odtwarzalności bit-exact (`GENERATOR_PARAMS.md` §4 pkt 4);
  świeży `Simulator` odtwarza tę samą sekwencję RNG (`e1`); IR audio identyczne co do bitu przez 5 renderów
  w dwóch świeżych instancjach `Simulator` (`GENERATOR_PARAMS.md`, uwaga w §5); rendering RGB/depth jest
  deterministyczny (99,98 % pikseli bit-identycznych z referencją VisualEchoes). Adaptacyjne `N` zależy
  od zmierzonego `sigma_1`, które jest funkcją tych samych deterministycznych renderów.
- **przeciw / niepewne:** (a) wyniki `e1`/determinizm RNG są oznaczone **[P]** w raporcie sesji — cytowane,
  niezweryfikowane ponownie w sesjach 07-28/29; (b) test dotyczył 5 renderów, nie ~12–76 tys. na scenę;
  (c) rozszerzenie jest kompilowane na `sm_90` i JIT-owane na Blackwell, więc inna GPU/sterownik mogą dać
  inne bity w części graficznej (RGB/depth) — niesprawdzone; (d) akumulacja `float32` zależy od kolejności
  sumowania, która jest stała w kodzie, ale kod mógłby się zmienić; (e) wznowienie (`--resume`) w środku
  sceny zmienia sekwencję RNG — `e1_checkpoint_boundary_merge` pokazuje, że statystycznie bezpiecznie
  (Wilcoxon p = 0,949), nie że bitowo identycznie. Wszystkie opublikowane pliki powstały w jednym
  przebiegu (`runs` ma 1 wpis), więc pełna regeneracja bez przerw jest odpowiednikiem.

### 3.3 Odpowiedź

- **SHA256 całego pliku `.h5` NIE jest powtarzalne** (znaczniki czasu w atrybutach i nagłówkach HDF5).
  `SHA256SUMS` w repozytorium służy do kontroli integralności pobrania, nie do weryfikacji regeneracji.
- **Bit-identyczność datasetów `echo`/`rgb`/`depth` jest oczekiwana przy `threadCount = 1` na tej samej
  konfiguracji, ale nie została potwierdzona pełną regeneracją.** Aby to potwierdzić: wygenerować jedną
  małą scenę (`office_1`, 576 próbek, ~30 min) i porównać datasety `echo`, `rgb`, `depth`, `position`,
  `n_total` bajt po bajcie (np. hash samych tablic, bez nagłówków) — dopiero wtedy można pisać „bit-identyczne".

---

## 4. Gotowy blok do karty („Odtworzenie zbioru")

````markdown
## Odtworzenie zbioru

Zbiór wygenerowano kodem z repozytorium pracy magisterskiej, w stanie odpowiadającym commitowi
`ccf211c4` (`ccf211c41eec93ee62278497ddbfcb46854fc372`). Dane powstały z drzewa roboczego tuż przed tym
commitem; pliki HDF5 nie zapisują hasha commita, a stan kodu ustalono z historii git i czasów generacji.
Oba warianty (`main`, `patched`) pochodzą z tego samego kodu i różnią się wyłącznie siatką sceny.

Środowisko: Python 3.10.14, numpy 1.26.4, scipy 1.15.3, h5py 3.14.0, habitat-sim 0.2.2
(baza `80f8e31140eaf50fe6c5ab488525ae1bdf250bd9` + lokalny patch `habitat-sim/local_changes.patch`),
rlr-audio-propagation `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`, jedna GPU z EGL.

```bash
# Replica w sound-spaces/data/scene_datasets/replica/<scena>/habitat/,
# lokalizacje w scenes_ve_metadata_locations/scene_observations_128.pkl (z VisualEchoes)
conda activate habitat

# main — jedna scena
python my-operations/generate_echo_dataset.py --scene office_1 --variant main
# patched — najpierw siatki z domkniętymi dziurami, potem scena
python my-operations/measurements/patch_scene_holes.py --all
python my-operations/generate_echo_dataset.py --scene office_2 --variant patched
```

Powtarzalność: generator używa `threadCount = 1`, więc ponowna generacja powinna dać te same tablice
`echo`/`rgb`/`depth`, ale pełna regeneracja nie została zweryfikowana; sumy SHA256 całych plików `.h5`
**nie** są powtarzalne (pliki zawierają znaczniki czasu, nazwę hosta i czasy renderów), więc `SHA256SUMS`
służy wyłącznie do sprawdzenia integralności pobrania.
````
