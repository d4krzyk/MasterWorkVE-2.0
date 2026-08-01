# Skrypty pomiarowe — materiał dowodowy

Jednorazowe pomiary, którymi rozstrzygnięto decyzje projektowe. **To nie jest kod
produkcyjny** — generator i diagnostyka żyją odpowiednio w `generate_echo_dataset.py`
i `diagnostics/`. Te skrypty istnieją, żeby dało się **odtworzyć i obronić** każdą liczbę
zapisaną w `docs/GENERATOR_PARAMS.md` i `docs/RAPORT_SESJI_2026-07-26_29.md`.

Każdy plik ma w docstringu: **pytanie**, na które odpowiada, **metodę**, **wynik z datą**
oraz odsyłacz do sekcji raportu i dokumentu.

## Mapa: co dowodzi czego

| skrypt | pytanie | GPU | raport |
|---|---|---|---|
| `agent_height_vs_pkl.py` | Czy `y` brać z `graph.pkl`, czy z `snap_point()`? | tak | §2.1 |
| `navmesh_offset_survey.py` | Jaka jest skala rozbieżności na wszystkich 1740 lokalizacjach? | **nie** | §2.1 |
| `audio_duplication_bench.py` | Ile kosztuje zdublowana symulacja audio i czy da się pominąć rendering wizualny? | tak | §2.2 |
| `audio_path_render.py` → `audio_path_analyse.py` | Czy 1 symulacja/render jest równoważna 2? | tak → nie | §2.2, §2.6 |
| `simulator_warmup.py` | Czy rozgrzewka jest własnością konstrukcji Simulatora, czy pozycji? | tak | §2.7 |
| `probe_estimator_accuracy.py` | Jaki jest rzeczywisty błąd sondy 8-renderowej? | **nie** | §2.6 |
| `signal_10deg_production.py` | Czy `SIGNAL_10DEG` trzyma się po poprawce `y`? | tak | §3.3 |
| `census_analysis.py` | Jaki jest pełny rozkład `N_raw` i czy `N_MAX` wystarcza? | **nie** | §2.3, §2.4 |
| `census_outlier_recheck.py` | Które przekroczenia `N_MAX` są prawdziwe? | tak | §2.5, §3.4 |
| `probe_discard_unittest.py` | Czy nadmiar sondy przy `N < 8` jest odrzucany? | **nie** | §3.3 pkt 2 dokumentu |
| `rt60_vs_sabine.py` | Czy pogłos z symulacji zgadza się z Sabine/Eyringiem? Które sceny są akustycznie zamknięte? | tak | §2.9, §2.10 |
| `soundspaces1_rt60.py` | Czy SoundSpaces 1.0 wykazuje tę samą sygnaturę braku sufitu? | **nie** | §2.10 |
| `ray_escape_survey.py` | Jaki ułamek promieni ucieka ze sceny w każdej z 1740 lokalizacji? | tak | §2.11 |
| `patch_scene_ceiling.py` → `ceiling_patch_rt60.py` | Czy domknięcie sceny sufitem zbliża RT60 do SoundSpaces 1.0? | nie → tak | §2.12 |
| `patch_scene_holes.py` | Gdzie są dziury w każdej scenie, jakiego są typu i czy da się je domknąć? | **nie** | §2.13 |
| `patch_material_sweep.py` | Jaki materiał łaty daje najlepszą zgodność z SS 1.0? *(dopasowanie, nie walidacja)* | tak | §2.13 |
| `cross_engine_rt60.py` | Czy SS 2.0 zgadza się z SS 1.0 tam, gdzie geometria jest cała — **bez dopasowanych parametrów**? | tak | §2.14 |

Sam census (`--probe-only`) jest częścią generatora, nie tego katalogu — jest trybem
produkcyjnym, nie pomiarem jednorazowym.

## Kolejność zależności

```
generate_echo_dataset.py --probe-only          ->  outputs/probe_census/*.csv
                                                        |
                                                        v
                                               census_analysis.py
                                               census_outlier_recheck.py

simulator_warmup.py    ->  outputs/measurements/warmup_specs.npz
                                                        |
                                                        v
                                          probe_estimator_accuracy.py

audio_path_render.py   ->  outputs/measurements/paths_specs.npz
                                                        |
                                                        v
                                             audio_path_analyse.py

patch_scene_holes.py   ->  outputs/patched_scenes/<scena>/habitat/mesh_semantic.ply
  (patch_scene_ceiling.py — poprzednik, tylko sufit, zostawiony bo na nim
   oparty jest wynik §2.12)                             |
                                    +-------------------+-------------------+
                                    v                                       v
                    ray_escape_survey.py --mesh ...              ceiling_patch_rt60.py
                    (kontrola: czy domkneta?)                    (czy RT60 blizej SS 1.0?)
                                                                            |
                                                                            v
                                                              patch_material_sweep.py
```

## Co gdzie ląduje

| katalog | zawartość | w gicie |
|---|---|---|
| `outputs/measurements/*.npz` | surowe spektrogramy (~210 MiB) | **nie** |
| `outputs/measurements/ray_escape/*.csv` | ucieczka promieni, 1740 lokalizacji (~256 KB) | **tak** (wyjątek w `.gitignore`) |
| `outputs/measurements/ceiling_patch/*.json` | wynik eksperymentu z sufitem (kilka KB) | **tak** (wyjątek w `.gitignore`) |
| `outputs/patched_scenes/` | załatane siatki, 84 MB na scenę | **nie** — odtwarza je `patch_scene_ceiling.py` |
| `outputs/probe_census/*.csv` | census 1740 lokalizacji (~120 KB) | **tak** (wyjątek w `.gitignore`) |
| `outputs/diagnose_rlr_noise_out/*.png` | wykresy | **tak** |
| `outputs/diagnose_rlr_noise_out/diagnostics_report.json` | wyniki eksperymentów | **tak** |

Pliki `.npz` są celowo poza gitem — to 210 MiB pośrednich danych, z których policzono
liczby zapisane w dokumentach. Jeśli mają być archiwizowane, warto obok zapisać sumy
kontrolne (`sha256sum outputs/measurements/*.npz`).

## Uwaga metodologiczna: dwa estymatory `sigma_1`

W projekcie występują **dwa** estymatory szumu i mylenie ich jest łatwe:

- **połówkowy** — `RMSE(A, B)/√2 · √(n/2)`, używany przez generator, bo musi być spójny
  z regułą wyznaczającą `N`. Ma **sufit dokładności ~4–6 %, niezależny od liczby renderów**
  (ograniczony korelacją przestrzenną spektrogramu, nie `n`).
- **wariancyjny** — `σ² = średnia po komórkach z Var po renderach`, `n−1` stopni swobody
  na komórkę, **poprawia się z `n`** (0.1–1.1 % przy 80 renderach).

Do wszystkich porównań dokładniejszych niż ~10 % używany jest wariancyjny. Pierwsze
podejście estymatorem połówkowym dało pozornie niepokojące +7 % i +18 % różnicy między
ścieżkami audio — okazało się artefaktem estymatora i rozgrzewki. Szczegóły:
`probe_estimator_accuracy.py` i raport §2.6.
