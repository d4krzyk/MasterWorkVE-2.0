# Czym dokładnie są „Model 1" i „Model 2" — słownik jednoznaczny

Dokument powstał **2026-08-13**, po tym jak okazało się, że numeracja w raportach jest odwrotna do
intuicji autora. **Nazwy nie zmieniają tego, co zostało policzone** — ale przy pisaniu pracy trzeba
mieć jedno źródło prawdy, więc jest nim ten plik.

Numeracja „Model 1 / Model 2" pochodzi z briefów sesji (10.08: *„BLOK 4 — Model 2, zadanie
pretekstowe orientacji"*). Utrzymuję ją w raportach dla spójności, ale **w pracy zalecam nazwy
opisowe** z ostatniej kolumny — są nieomylne.

---

## Tabela rozstrzygająca

| nazwa w raportach | co dostaje na wejściu | co przewiduje | zalecana nazwa w pracy |
|---|---|---|---|
| **Model 1, wersja `full`** | obraz RGB **+** echo | głębię | **sieć głębi (pełna)** |
| **Model 1, wersja `echo2depth`** | **samo echo** | głębię | **sieć głębi (tylko echo)** |
| **Model 2** | para: widok z kąta *i* **+** echo z kąta *j* | **obrót** *j − i* | **zadanie pretekstowe orientacji** |
| **transfer** | sam obraz RGB | głębię | **zadanie docelowe (transfer)** |

W kodzie odpowiadają temu katalogi `ml/depth_model/` i `ml/pretext_model/`, które są nazwane
opisowo i **nie mają tej dwuznaczności**.

---

## Model 1 — sieć głębi (to jest główny eksperyment pracy)

Architektura **Paridy** (*Beyond Image to Depth*, CVPR 2021), niezmieniona co do znaku. Zmienną
niezależną jest **gęstość kątowa zbioru treningowego** (4 / 6 / 9 / 12 / 18 / 36 orientacji).

### Wersja `full` — obraz + echo → głębia

Cztery podsieci Paridy:

| podsieć | parametry | rola |
|---|---|---|
| `RGBDepthNet` | 16 658 561 | U-Net: obraz → głębia |
| `SimpleAudioDepthNet` | 8 984 073 | Echo-Net: spektrogram → głębia |
| `attentionNet` | 279 581 505 | fuzja obu strumieni |
| `MaterialPropertyNet` | 11 694 642 | ResNet-18, klasyfikacja materiału |
| **razem** | **316 918 781** | |

Warunki: `A` (4 kąty), `B` (36), `D` (4 losowane), `SE` (echo permutowane), `PA`/`PB`/`PD` (geometria
`patched`).

### Wersja `echo2depth` — samo echo → głębia

**Tylko** `SimpleAudioDepthNet` (8 984 073 parametry), wyjęty z fuzji. Bez obrazu, bez materiału,
bez uwagi. Przedrostek **`E`**: `EA`, `EB`, `ED`, `ESE`, `EK6..EK18`, `EPA`/`EPB`/`EPD`.

**Po co ta wersja istnieje:** w pełnym modelu obraz RGB niesie ~91 % informacji o głębi, więc efekt
gęstości kątowej echa tonie w priorze wizualnym. Wersja bez obrazu jest **najczystszym testem
hipotezy pracy** — cała informacja pochodzi wtedy z echa.

**To rozróżnienie jest źródłem pozornej sprzeczności w raportach.** Efekt gęstości wynosi **0,147**
w `echo2depth` i **0,018** w `full`. Oba są dodatnie; różnią się siłą, bo w pełnym modelu poprawa
echa działa na 9 % całości, nie na 100 %.

---

## Model 2 — zadanie pretekstowe orientacji (**nie** przewiduje głębi)

Sieć dostaje **parę z tej samej lokalizacji**: widok z orientacji *i* oraz echo z orientacji *j*.
Ma odgadnąć **o ile stopni agent jest obrócony**, czyli klasę *(j − i)*. Głębia w ogóle nie
występuje.

Architektura wg suplementu Gao §I: enkoder `RGBDepthNet` (5 warstw, 128×128×3 → 4×4×512) +
`SimpleAudioDepthNet` bez dekodera + konkatenacja → warstwa w pełni połączona → D = 128 → K klas,
płaska cross-entropy. **Bez `MaterialPropertyNet`.** Razem 25 733 446 parametrów.

### Transfer — to, po co Model 2 w ogóle powstał

Bierzemy **wytrenowany koder wizualny** z zadania pretekstowego i wstawiamy go jako inicjalizację do
sieci przewidującej **głębię z samego obrazu, bez echa**. Reszta sieci startuje losowo.

**„Transfer" = przeniesienie wag kodera**, nic więcej. Porównanie: `Scratch` (koder losowy) wobec
`pretrening K=4 / 12 / 36 / 36@16par`.

**Uwaga na pułapkę interpretacyjną:** to, że Model 2 dobrze rozwiązuje **swoje** zadanie
(MAAE 25,13° wobec 90° losowego przy K=36), **nie znaczy**, że transfer pomoże w głębi. To dwie
różne rzeczy i zmierzyliśmy je osobno — patrz `RAPORT_SESJI_2026-08-13.md` §5 i §5.1.

---

## Gdzie jest `MaterialPropertyNet`

| model | MaterialNet | dlaczego |
|---|---|---|
| Model 1 `full` | **jest** | wierność architekturze referencyjnej — usunięcie unieważniłoby zdanie „sieć jest dokładnie ta opublikowana" |
| Model 1 `echo2depth` | nie ma | brak obrazu, więc materiał i uwaga nie mają czego łączyć |
| Model 2 (pretekst) | nie ma | suplement Gao §I go nie przewiduje |
| transfer | nie ma | sam `RGBDepthNet` |

**Sprostowanie [Z], 2026-08-13.** `ModelBuilder.build_material_property()` wołane bez `init_weights`
— czyli tak, jak my je wołamy — idzie gałęzią **`resnet18(pretrained=False)`**. `MaterialNet`
startuje więc z **wag losowych, nie z ImageNetu**. Jest to zgodne z kodem Paridy przy tym
wywołaniu.

Konsekwencja: normalizacja obrazu statystykami ImageNetu (`mean` 0,485/0,456/0,406) **nie jest**
uzasadniona pretrenowanym ResNetem, jak twierdził wcześniejszy komentarz w
`ml/dataset/echo_h5_dataset.py`. Prawdziwe uzasadnienie: **tak normalizuje Parida**, a my
odtwarzamy jego potok co do znaku. Wartości bez zmian; poprawione zostało wyłącznie uzasadnienie,
żeby błędne nie trafiło do rozdziału o implementacji.

---

## Skrócony słownik identyfikatorów warunków

Pełna wersja w `my-operations/ml/README.md` §4.

| element | znaczenie |
|---|---|
| `A` / `B` / `D` | 4 kąty kardynalne / 36 kątów / 4 kąty losowane per lokalizacja |
| `E…` | wersja **`echo2depth`** (samo echo) |
| `P…` | geometria **`patched`** (dziury domknięte) |
| `SE` / `ESE` | echo **permutowane** — kontrola „ile w ogóle wnosi echo" |
| `EK6`…`EK18` | stały budżet 4 próbek/lokalizację, kąty z siatki K |

`EPD` = `echo2depth` + `patched` + 4 kąty losowane.
