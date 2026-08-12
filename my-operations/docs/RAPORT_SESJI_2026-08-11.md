# Raport sesji — 2026-08-11: domknięcie protokołu, bramka pełnego modelu, potwierdzenie wyniku

Kontynuacja `RAPORT_SESJI_2026-08-10.md`. Trzy cele: naprawić protokół walidacji (blokował
macierz), zmierzyć wkład echa w **pełnym** modelu (decyduje o grupie `glowne`), potwierdzić wynik
grupy `echo` na 3 ziarnach. Plus poprawki interpretacyjne i eksport liczb do pisania tekstu.

Statusy: **[Z]** zmierzone · **[Z-]** zmierzone z zastrzeżeniem · **[W]** wywnioskowane z kodu ·
**[X]** nie sprawdzone.

---

## 0. Budżet dysku **[Z]**

Sprawdzone **przed** uruchomieniem czegokolwiek. `outputs/ml/disk_budget.json`.

| pozycja | GB |
|---|---|
| wolne na `/home` | **227,1** |
| zajęte: `echoes_36deg` + `_patched` | 26,0 |
| zajęte: `outputs/ml/` | 0,55 |
| na przebieg `echo2depth` (2× wagi + checkpoint + npz) | 0,168 |
| na przebieg pełnego modelu | 5,904 |
| **ta sesja** (§2: 2 pełne, §3: 12 `echo2depth`) | **13,82** |
| pozostała macierz po tej sesji | 187,9 |
| **wszystko + zapas 20 GB** | **221,7** |

**Werdykt sesji: OK** (13,82 + 20 ≪ 227,1).
**Werdykt całej macierzy: mieści się, ale margines to 5,4 GB.**

Rozbicie pozostałej macierzy: `krzywa` 70,8 · `geometria` 53,1 · `glowne` 47,2 · `bramka` 12,1 ·
`krzywa_staly` 2,0 · `geometria_echo` 1,5 · `echo` 1,0.

**Propozycje (nie wykonane — decyzja autora), gdyby margines okazał się za mały:**
kasować `checkpoint.pt` po przebiegach z `finished=true` (zwalnia 3/5 miejsca każdego przebiegu,
~120 GB na całej macierzy); zapisywać wagi archiwalne w `float16`; pomijać `best4_*.pth` tam, gdzie
`best_step_same = true` (§1.1 — dzieje się automatycznie).

---

## 1. Protokół walidacji — decyzja §4.7 wprowadzona **[Z]**

**Wszystkie warunki walidują teraz na `angle_subset="all"` (36 kątów)**, niezależnie od własnego
podzbioru treningowego. `train_condition.VAL_ANGLE_SUBSET = "all"`.

Argument rozstrzygający, do wpisania w pracy: **zbiór testowy jest już wspólny dla wszystkich
warunków.** Nikt nie proponuje testować `cardinal` na 4 kątach, a `all` na 36 — bo to oczywiście
unieważniłoby porównanie. Walidacja i test to **ta sama kategoria**: dane odłożone, reprezentujące
rozkład, na którym model ma działać. Traktowanie ich różnie było **dryfem implementacyjnym**, nie
decyzją projektową.

### 1.1 Dwie krzywe z jednego przelotu **[Z]**

`val@4` jest podzbiorem `val@36`, więc liczy się **z tych samych predykcji** — przez wybór wierszy
o kącie kardynalnym z tabeli statystyk per próbka. **Zero dodatkowego przelotu.**

`status.json` zawiera teraz: `best_val36_rmse` / `best_step_val36` (**podstawowe**, kryterium dla
całej macierzy), `best_val4_rmse` / `best_step_val4` (kolumna odporności) oraz `best_step_same`.
Wagi: `best_<net>.pth` i `best4_<net>.pth`; przy `best_step_same = true` drugi komplet nie powstaje.

Sprawdzone przebiegiem kontrolnym (`EA`, 20 kroków): walidacja poszła na 6 588 próbkach zamiast 732,
oba kryteria zapisane, `best_step_same = true`, oba pliki wag na dysku.

Koszt dyskowy: pełny model **+1,18 GB na przebieg**, czyli **+10,6 GB na 9 przebiegów `glowne`**
(zgodnie z przewidywaniem). `echo2depth` +0,034 GB/przebieg. `experiments.py::disk_budget` liczy
teraz 5× parametry zamiast 4×; cała macierz **200,2 GB** (było 160,6).

### 1.2 Zastrzeżenie **[Z-]**

Przy `val@36` warunek `cardinal` wybiera checkpoint, **korzystając z rozkładu kątowego, którego nie
ma w swoim zbiorze treningowym**. Nie unieważnia to porównania — pytanie pracy brzmi „ile agent
traci, trenując na 4 kątach, skoro działać musi pod dowolnym", a **nie** „ile traci praktyk mający
dostęp wyłącznie do 4 orientacji, także przy wyborze modelu". Ale te dwa pytania **trzeba w tekście
nazwać i rozróżnić**; kolumna `val@4` istnieje właśnie po to, żeby zmierzyć, ile ta różnica wynosi.

---

## 2. Bramka pełnego modelu — `c_full` zmierzone, decyzja o grupie `glowne` **[Z]**

`SE` i `B`, pełny model, `--fast-bilinear`, ziarno 0, 40 000 kroków, walidacja wg protokołu z §1.
Oba przebiegi: `best_step = 38 000`, `best_step_same = true`, **`budget_ceiling_warning = false`**
— nowa heurystyka plateau (§3.4 poprzedniego raportu) działa też na pełnym modelu.

| | `test@36` | `test@4` | krawędzie | gładkie |
|---|---|---|---|---|
| `B` (echo prawdziwe) | **0,24205** | 0,24690 | 0,52703 | 0,17198 |
| `SE` (echo permutowane) | **0,26433** | 0,27098 | 0,54459 | 0,19966 |

Bootstrap sparowany po 183 lokalizacjach, 2 000 losowań:

| warstwa | `c_full` = ΔRMSE | 95 % CI | istotne |
|---|---|---|---|
| **całość** | **+0,02228** | **[+0,01840; +0,02643]** | tak |
| krawędzie | +0,01756 | [+0,01227; +0,02317] | tak |
| gładkie | +0,02768 | [+0,02288; +0,03250] | tak |

### Zastosowanie reguły zapisanej PRZED pomiarem

```
f_ang = (ED − EA) / (ESE − EB) = 0,13957 / 0,58808 = 0,2373      (z grupy echo, ziarno 0)
bound = f_ang × c_full        = 0,2373 × 0,02228   = 0,00529
```

Podłoga szumu frameworka: 0,00232–0,00732. **`bound` to 0,72–2,28× podłogi**, czyli mieści się
w szumie. Próg praktyczny (`c_full > 0,063`, żeby `bound` sięgnął 3× podłogi) **nie został
przekroczony** — `c_full` = 0,0223, czyli ~2,8× za mało.

| `bound` | próg reguły | wynik |
|---|---|---|
| 0,00529 | < 0,015 | **trzeci wiersz tabeli** |

### DECYZJA: **degradacja grupy `glowne`**

**Grupa `glowne` idzie po 1 ziarnie — dla kompletności protokołu, nie jako źródło dowodu.**
Ciężar dowodu przechodzi na grupę **`echo`** (§3, 3 ziarna) i **Model 2**. Praca ma stwierdzić
wprost: **pełny model nie rozdziela efektu gęstości kątowej**, bo przewidywany efekt leży poniżej
własnej podłogi szumu obliczeń.

Oszczędność: 9 → 3 przebiegi w `glowne`, czyli **−5,2 h GPU i −35,4 GB dysku**.

### Co ta liczba znaczy — i czego nie znaczy

**Wkład echa w pełnym modelu jest 26,4× mniejszy niż w `echo2depth`** (0,0223 wobec 0,5881).
Prior wizualny niesie prawie całą informację o głębi; echo dokłada margines.

**Kontrola poprawności rzędu wielkości:** względny wkład echa u nas to **9,2 %** (0,0223 / 0,2420),
u Gao **7,5 %** (0,374 → 0,346). Te dwie liczby pochodzą z innych silników akustycznych i nie wolno
ich zestawiać w jednej kolumnie — ale ich zgodność co do rzędu wielkości jest **niezależnym
potwierdzeniem, że nasz pełny model zachowuje się jak opublikowany**, a nie że echo zostało
podłączone wadliwie. Gdyby `c_full` wyszło np. 0,001, pierwszym podejrzanym byłby błąd w potoku.

**[Z-] Zastrzeżenie do `bound` — musi iść razem z liczbą.** `bound` jest **heurystyką alokacji
budżetu, nie twierdzeniem naukowym**. Multiplikatywność frakcji orientacyjnej między architekturami
nie jest niczym gwarantowana: `f_ang` zmierzono na `echo2depth` i przeniesiono na pełny model **bez
dowodu**. Liczba służy wyłącznie do decyzji „ile ziaren kupić" i **nie może trafić do pracy jako
wynik**. Faktyczny efekt gęstości w pełnym modelu zmierzą warunki `A`/`D` — po 1 ziarnie, z jawnie
zapisanym ograniczeniem mocy.

---

## 3. Grupa `echo` na 3 ziarnach — wynik potwierdzony **[Z]**

12 przebiegów (`EA`/`EB`/`ED`/`ESE` × ziarna 0/1/2), **wszystkie po wprowadzeniu §1**, więc liczby
z §5.6 poprzedniego raportu zostają zastąpione. Ewaluacja na wspólnym `test@36` (6 588 próbek,
183 lokalizacje).

**Dwa źródła zmienności, konsekwentnie rozdzielone:** `sd` po **ziarnach** (inicjalizacja wag,
niedeterminizm obliczeń) i `CI` bootstrapu po **lokalizacjach** (zmienność zbioru testowego). To nie
są te same wielkości i nie wolno ich mylić ani łączyć.

### RMSE na `test@36` po ziarnach

| warunek | ziarno 0 | ziarno 1 | ziarno 2 | **średnia** | **sd (ziarna)** |
|---|---|---|---|---|---|
| `EA` `cardinal` | 0,77895 | 0,79910 | 0,79508 | **0,79104** | 0,01066 |
| `ED` `random_4` | 0,64695 | 0,64232 | 0,64369 | **0,64432** | 0,00238 |
| `EB` `all` | 0,58311 | 0,58519 | 0,57839 | **0,58223** | 0,00348 |
| `ESE` permutowane | 1,16468 | 1,16533 | 1,15876 | **1,16292** | 0,00362 |

Rozrzut po ziarnach dla `ED`/`EB`/`ESE` (0,0024–0,0036) mieści się w zmierzonej podłodze szumu
frameworka (0,0023–0,0073). **`EA` ma rozrzut 3–4× większy** (0,01066) — warunek o najrzadszym
pokryciu kątowym jest najbardziej wrażliwy na inicjalizację, co jest osobną obserwacją wartą
odnotowania przy planowaniu liczby ziaren dla warunków rzadkich.

### Rozkład efektu — 3 ziarna

| składowa | średnia ± sd (ziarna) | CI po lokalizacjach (ziarno 0) | istotne we wszystkich 3 |
|---|---|---|---|
| **gęstość kątowa (D − A)** | **+0,14672 ± 0,01303** | [+0,11231; +0,15178] | tak |
| ilość danych (B − D) | +0,06209 ± 0,00435 | [+0,05124; +0,07881] | tak |
| łączny (B − A) | +0,20882 ± 0,01132 | [+0,16971; +0,22545] | tak |
| wkład echa (ESE − EB) | **+0,58070 ± 0,00076** | [+0,54044; +0,62404] | tak |

**Udział gęstości kątowej w efekcie łącznym: 70,2 % ± 3,0 pp** (na 1 ziarnie i starym protokole
wychodziło 62,1 %). Gęstość jest **2,36×** większa niż efekt ilości danych — poprzednio 1,64×.
Kierunek wniosku bez zmian, **siła większa**.

Wkład echa jest wyjątkowo stabilny między ziarnami (sd 0,00076, czyli **poniżej podłogi szumu**) —
to wielkość wyznaczona przez informację w danych, nie przez przebieg optymalizacji.

### Wpływ zmiany protokołu na lukę generalizacji kątowej

Poprzedni raport przewidywał, że stary układ (walidacja na własnym podzbiorze) **zawyżał** lukę
warunku `EA`, bo checkpoint był wybierany krótkowzrocznie na 4 kątach. **Przewidywanie potwierdzone
co do kierunku:**

| | luka `EA` (0° → 40°) |
|---|---|
| stary protokół, ziarno 0 | +0,35540 (61,50 %) |
| **nowy protokół, 3 ziarna** | **+0,31477 ± 0,02860 (52,84 %)** |
| różnica | **−0,04063** |

Luka pozostaje **duża i monotoniczna we wszystkich trzech ziarnach**, ale jest o ~11 % (względnie)
mniejsza, niż raportowano. To jest dokładnie ten rodzaj korekty, dla którego zmiana protokołu była
konieczna.

Krzywa (średnia ± sd po ziarnach):

| odległość od siatki treningowej | RMSE |
|---|---|
| 0° | 0,59626 ± 0,00822 |
| 10° | 0,66296 ± 0,00227 |
| 20° | 0,78249 ± 0,00870 |
| 30° | 0,87034 ± 0,01575 |
| 40° | 0,91103 ± 0,02190 |

### Tabela luki per warunek — potwierdzona na 3 ziarnach

| warunek | luka `test@36` − `test@4` |
|---|---|
| **`EA`** | **+0,19478 ± 0,01763** |
| `ED` | +0,00691 ± 0,00297 |
| `EB` | +0,00384 ± 0,00201 |
| `ESE` | +0,00395 ± 0,00019 |

Kontrola mechanizmu z §4.2 przechodzi na 3 ziarnach: lukę ma **wyłącznie** warunek bez pokrycia
kątowego, pozostałe trzy mają 0,004–0,007, czyli na poziomie podłogi szumu.

---

## 4. Poprawki do interpretacji liczb z §5.6 poprzedniego raportu

### 4.1 Porównanie sparowane zamiast „przez dwa zbiory testowe" **[Z]**

Poprzedni raport zestawiał `EA` na `test@4` (0,57792) z `EB` na `test@36` (0,57783) i nazywał je
identycznymi. To **dwa różne zbiory testowe** (732 wobec 6 588 próbek) — porównanie podatne na
zarzut doboru. Właściwa liczba stała obok, w tej samej tabeli.

Bootstrap sparowany po lokalizacjach, ograniczony do `test@4` (**identyczne 732 próbki**, 183
lokalizacje). Policzone **dwa razy** — na starych checkpointach i na nowych, po zmianie protokołu
z §1 — bo wynik okazał się od protokołu zależny:

| protokół | ΔRMSE (`EA` − `EB`) na `test@4` | werdykt |
|---|---|---|
| stary (walidacja na 4 kątach), ziarno 0 | +0,00150, CI [−0,01325; +0,01731] | obejmuje zero |
| **nowy (walidacja na 36 kątach), 3 ziarna** | **+0,01787 ± 0,01128** | **istotne w 2 z 3 ziaren** |

**Twierdzenie z poprzedniego raportu wymaga korekty i tu ją wprowadzam.** Zdanie „model trenowany na
4 orientacjach jest na tych 4 orientacjach dokładnie tak samo dobry" było prawdziwe **wyłącznie pod
starym protokołem** — i to nie przypadkiem: tam checkpoint warunku `EA` był wybierany dokładnie na
tej metryce, na której go potem porównywano. Po usunięciu tego sprzężenia `EA` okazuje się na
kątach kardynalnych **mierzalnie, choć nieznacznie gorszy**.

Poprawna wersja twierdzenia, na 3 ziarnach i nowym protokole:

| gdzie | `EA` − `EB` |
|---|---|
| na 4 kątach, które `EA` widział | **+0,01787** |
| na wszystkich 36 kątach | **+0,20882** |

**91,4 % kary warunku `EA` powstaje na orientacjach, których nie widział**, a nie na gorszym
nauczeniu tych, które widział. Wniosek jakościowy się utrzymuje — baseline Gao głównie **nie
pokrywa przestrzeni orientacji**, a nie „uczy się gorzej" — ale nie wolno pisać, że modele są na
kątach treningowych **identyczne**. Różnica jest niezerowa i wynosi ~2,4× podłogi szumu.

Rozbicie na warstwy (nowy protokół, 3 ziarna): krawędzie −0,01871 ± 0,03576 (istotne w 1/3),
gładkie +0,03116 ± 0,00911 (istotne w 3/3). Kierunek specjalizacji `EA` w krawędziach, zauważony
przy starych checkpointach, **przestaje być istotny** po zmianie protokołu — czyli był w dużej
mierze artefaktem wyboru checkpointu, a nie własnością modelu.

### 4.2 Tabela luki per warunek — kontrola mechanizmu **[Z]**

Zero GPU, samo grupowanie istniejących predykcji. Wersja na 3 ziarnach i nowym protokole jest
w §3; poniżej wersja z ziarna 0 i starego protokołu, dla porównania.
`outputs/ml/echo_ablation/gap_table_seed0.json`.

| warunek | `test@36` | `test@4` | **luka** | pokrycie kątowe zbioru treningowego |
|---|---|---|---|---|
| **`EA`** | 0,80265 | 0,57792 | **+0,22473** | **4 kąty kardynalne** |
| `ED` | 0,66308 | 0,66215 | +0,00093 | 36 kątów (globalnie, 4 na lokalizację) |
| `EB` | 0,57783 | 0,57642 | +0,00141 | 36 kątów |
| `ESE` | 1,16591 | 1,16192 | +0,00399 | 36 kątów (echo permutowane) |

**Lukę ma wyłącznie warunek bez pokrycia kątowego.** `ED` ma tyle samo próbek co `EA` (5 496) i tak
samo szybko zbiega, a luki nie ma. `ESE` nie ma użytecznego sygnału i luki też nie ma. To wyklucza
trzy najbardziej oczywiste alternatywne wyjaśnienia kary warunku `EA`: mniejszy zbiór, gorszą
zbieżność i artefakt ewaluacji. Zostaje **brak pokrycia orientacji**.

### 4.3 Usunięte zdanie o sumowaniu się składowych

Zdanie „składowe sumują się dokładnie: 0,13957 + 0,08525 = 0,22482" **znika**. To tożsamość
algebraiczna `(D−A) + (B−D) = B−A`, prawdziwa niezależnie od danych; podana jako potwierdzenie
sugerowałaby niezrozumienie własnego pomiaru.

Wynikiem są **rozmiary składowych i ich stosunek**: gęstość kątowa **0,1467 ± 0,0130** wobec ilości danych
**0,0621 ± 0,0044**, czyli **2,36×** (§3, 3 ziarna), przy czym obie są istotne we wszystkich trzech
ziarnach i obie leżą wielokrotnie nad podłogą szumu.

### 4.4 Wniosek praktyczny — wyróżniony **[Z]**

> **Korzyść z gęstego próbkowania kątowego nie wymaga gęstego próbkowania w każdej lokalizacji —
> wymaga, żeby zbiór jako całość pokrywał przestrzeń orientacji.**

Warunek `D` to **4 rendery na lokalizację**, czyli dokładnie koszt protokołu Gao, a odzyskuje
**70,2 % ± 3,0 pp** przewagi pełnych 36 orientacji. Kto ma budżet 4 renderów na pozycję, powinien **losować
kąty zamiast przybijać je do kierunków kardynalnych** — i dostaje prawie dwie trzecie efektu za
1/9 kosztu generowania.

Rekomendacja jest bezpośrednio wykonalna i nikt nie mógł jej wcześniej wypowiedzieć, bo nikt nie
miał 36 orientacji do podpróbkowania.

---

## 5. Maska ścisła — kontrola wrażliwości **[Z] (implementacja) / [X] (Δ na obu maskach)**

Trzy maski, `DatasetConfig.mask_mode`:

| maska | definicja |
|---|---|
| `full` | `depth_gt != 0` — wewnątrz wariantu |
| `intersection` | ważne w obu wariantach — **podstawowa dla `main` vs `patched`** |
| `strict` | przecięcie **minus** piksele o różnej wartości głębi |

Zweryfikowane na 400 próbkach zbioru testowego (`outputs/ml/mask_check/mask_check.json`):

| | % kadru |
|---|---|
| ważne w `patched` (`full`) | 99,613 |
| `intersection` | 90,844 |
| `strict` | 89,245 |
| **różnica = piksele „zmienione, a już ważne"** | **1,599** |

Per scena różnica przecięcie − ścisła: `frl_apartment_5` **3,443 %** (poprzedni raport §2.2: 3,28 %),
`apartment_2` **0,652 %** (§2.2: 0,94 %), **`office_4` dokładnie 0,000 %** — scena szczelna, ten sam
plik w obu wariantach, więc zero różnic. Rzędy się zgadzają; rozbieżność wobec §2.2 to inna próbka
(400 losowych próbek wobec skanu co 6. wiersza).

**Narzut `mask_variant`** — §7 poprzedniego raportu wymieniał to jako niezmierzone:
bez maski **1,94 ms/próbkę**, `intersection` **2,65 ms (+37 %)**, `strict` **2,39 ms (+23 %)**.
Dotyczy wyłącznie ewaluacji (jeden przelot po zbiorze testowym), więc w skali przebiegu jest
nieistotne.

**[X] Czego to NIE rozstrzyga.** Pytanie z §2.2 brzmi „czy Δ(`main` vs `patched`) wychodzi to samo
na obu maskach". Odpowiedź wymaga modelu **trenowanego na `patched`**, a grupa `geometria` jest
w tej sesji zakazana. Zastrzeżenie z §2.2 **zostaje** w rozdziale o ograniczeniach do czasu
uruchomienia `EPA/EPB/EPD` (0,4 h) — narzędzie jest gotowe, brakuje wyłącznie checkpointu.

---

## 6. Eksport liczb do pisania pracy **[Z]**

`my-operations/ml/thesis_numbers.py` (zero GPU) zbiera **każdą zmierzoną liczbę** z trzech raportów
sesji i plików dowodowych, i produkuje:

- `outputs/ml/thesis_numbers.json` — źródło maszynowe
- `my-operations/docs/LICZBY_DO_PRACY.md` — do czytania przy pisaniu

**68 pozycji**, każda z wartością, jednostką, statusem `[Z]/[Z-]/[W]`, plikiem dowodowym i sekcją
raportu, w której jest omówiona:

| grupa | pozycji |
|---|---|
| 1. Zbiór danych | 13 |
| 2. Charakterystyka silnika akustycznego | 9 |
| 3. Geometria `main` vs `patched` | 11 |
| 4. Determinizm i wydajność | 10 |
| 5. Wyniki grupy `echo` | 10 |
| 6. Budżet obliczeniowy i dyskowy | 8 |
| **7. Odniesienia z literatury** | **7** |

Grupa 7 jest **wydzielona i opatrzona ostrzeżeniem w treści dokumentu**: liczby Gao i Paridy nie są
naszymi pomiarami, silnik akustyczny jest inny, więc służą wyłącznie do sprawdzenia **porządku**
warunków i **rzędu wielkości** efektu — nigdy do zestawienia w jednej kolumnie z naszymi wynikami.

Na końcu dokumentu: **9 liczb, których jeszcze nie ma**, każda z odsyłaczem do warunku, który ją da,
i uzasadnieniem, po co jest potrzebna — żeby autor wiedział, gdzie w tekście zostawić lukę, zamiast
odkrywać brak przy składaniu tabeli.

Plik regeneruje się jedną komendą, więc po każdym kolejnym przebiegu wystarczy uruchomić go
ponownie; nie należy edytować `LICZBY_DO_PRACY.md` ręcznie.

---

## 7. Czego **NIE** sprawdzono / nie zrobiono **[X]**

- **`ESA` (permutacja kątowa wewnątrz lokalizacji) — NIE zaimplementowana.** Zgodnie z kolejnością
  poświęcania z briefu (§7 pierwsza do odpuszczenia). Przestała być bramką, bo frakcję orientacyjną
  daje już `(ED − EA) / (ESE − EB)`. Zostaje jako samodzielny pomiar „czy orientacja echa niesie
  informację **per próbka**". Uwaga przy implementacji: derangement liczy się w **1 374 niezależnych
  grupach po 36**, więc ryzyko błędu bijekcji jest wyższe niż w §3.3 poprzedniego raportu — użyć
  zamian skalarnych i jawnej kontroli, tak jak tam.
- **Δ(`main` vs `patched`) na masce przecięcia i ścisłej — nie policzone.** Maski działają
  (§5), brakuje modelu trenowanego na `patched`. Grupa `geometria` zakazana w tej sesji.
- **Grupa `glowne` nie uruchomiona** — decyzja z §2 mówi *ile* ziaren, nie *że już*.
  Efekt gęstości w pełnym modelu (`D − A`) pozostaje niezmierzony.
- **Grupy `krzywa`, `krzywa_staly`, `geometria`, `geometria_echo` nieuruchomione** (zgodnie z zakazem).
- **Model 2 nie trenowany** dłużej niż 60 kroków przebiegu dymnego (poprzednia sesja).
- **`c_full` z jednego ziarna.** CI pochodzi z bootstrapu po lokalizacjach, nie z rozrzutu po
  ziarnach. Dla `EA` zmierzony rozrzut po ziarnach (0,0107) jest 3–4× większy niż dla pozostałych
  warunków — nie wiadomo, jak duży jest dla pełnego modelu.
- **Zmiana protokołu nie została przetestowana na pełnym modelu pod kątem `best_step_val4`** —
  w `SE` i `B` oba kryteria wskazały ten sam krok, więc kolumna odporności nie miała okazji się
  rozejść. Nie wiadomo, czy kiedykolwiek się rozejdzie.
- **Nie zmierzono, ile `best_step_same = true` oszczędza realnie** — budżet 5× parametry zakłada
  pesymistycznie, że oba komplety wag zawsze powstają.
- **`evaluate.py --compare` nie waliduje, że oba przebiegi użyły tego samego protokołu walidacji.**
  Zauważone, nie naprawione: porównanie checkpointu sprzed §1 z checkpointem po §1 przejdzie bez
  ostrzeżenia. Do dopisania jednej asercji na `status.json: val_angle_subset`.

## 8. Do dokończenia w następnej sesji

W kolejności:

1. **`EPA`/`EPB`/`EPD`, ziarno 0** (0,4 h GPU) → zamyka §5: Δ na masce przecięcia i ścisłej,
   czyli usuwa zastrzeżenie o pikselach „zmienionych a ważnych" z rozdziału o ograniczeniach.
   Przy okazji daje sondę `office_4` z §2.5 poprzedniego raportu.
2. **Grupa `glowne` (`A`, `B`, `D`), 1 ziarno** (~2,6 h GPU) — `B` jest już policzone, więc
   realnie `A` i `D`. Efekt gęstości w pełnym modelu, z jawnym ograniczeniem mocy z §2.
3. **Model 2: pretrening K ∈ {4, 12, 36} + K=36@16par, potem `transfer.py` × 5 inicjalizacji.**
   To jest teza z postera i największa niewiadoma pracy. `pretext/summarize.py` złoży tabele.
4. **Grupa `krzywa_staly` (`EK6/EK9/EK12/EK18`, 3 ziarna, 1,6 h)** — kształt zależności od gęstości
   przy stałej liczności; tańsza i ciekawsza niż `krzywa`.
5. `ESA`, jeśli będzie czas (§7).

Po każdym z tych kroków uruchomić `python my-operations/ml/thesis_numbers.py` — plik
`LICZBY_DO_PRACY.md` odświeży się sam, razem z listą liczb wciąż brakujących.
