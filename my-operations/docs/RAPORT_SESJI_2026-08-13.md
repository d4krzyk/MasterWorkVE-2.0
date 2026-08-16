# Raport sesji — 2026-08-13: wyniki kolejki nocnej

Kolejka `ml_ctl.py` przeliczyła **47 kroków w 10 h 42 min, 0 błędów, 0 pominiętych**.
Wolne miejsce na końcu: 197,2 GB. Log zbiorczy: `outputs/ml/logs/ml_ctl_2026-08-13_0129.md`.

Trzy wyniki: **dwa mocne pozytywne i jeden negatywny**. Negatywny opisany jest tak samo dokładnie
jak pozostałe — jest wynikiem, nie porażką.

Statusy: **[Z]** zmierzone · **[Z-]** z zastrzeżeniem · **[W]** wywnioskowane · **[X]** niesprawdzone.
Plik dowodowy całości: `outputs/ml/echo_ablation/final_results_2026-08-13.json`.

---

## 0. Nazewnictwo modeli — sprostowanie [Z]

Numeracja „Model 1 / Model 2" w raportach jest **odwrotna do intuicji autora**. Powstał
`docs/MODELE.md` jako **jedno źródło prawdy**; w pracy zalecane są nazwy opisowe:

| nazwa w raportach | wejście | przewiduje | zalecana nazwa w pracy |
|---|---|---|---|
| Model 1, wersja `full` | obraz **+** echo | głębię | **sieć głębi (pełna)** |
| Model 1, wersja `echo2depth` | **samo echo** | głębię | **sieć głębi (tylko echo)** |
| Model 2 | para (widok *i*, echo *j*) | **obrót** *j−i* | **zadanie pretekstowe orientacji** |
| transfer | sam obraz | głębię | **zadanie docelowe (transfer)** |

**Nic nie zostało policzone inaczej** — rozjazd jest wyłącznie w etykietach. Katalogi w kodzie
(`ml/depth_model/`, `ml/pretext_model/`) są nazwane opisowo i tej dwuznaczności nie mają.

Rozróżnienie `full` / `echo2depth` jest źródłem pozornej sprzeczności w §1 i §2: efekt gęstości
wynosi **0,147** w `echo2depth` i **0,018** w `full`. Oba dodatnie — w pełnym modelu poprawa echa
działa na ~9 % informacji, nie na 100 %.

### Gdzie jest `MaterialPropertyNet` [Z]

Wyłącznie w wersji `full` (wierność architekturze referencyjnej). Nie ma go w `echo2depth`,
w zadaniu pretekstowym ani w transferze.

**Sprostowanie:** `build_material_property()` wołane bez `init_weights` idzie gałęzią
`resnet18(pretrained=False)` — `MaterialNet` startuje z **wag losowych, nie z ImageNetu**. Zgodnie
z kodem Paridy. Konsekwencja: normalizacja obrazu statystykami ImageNetu **nie jest** uzasadniona
pretrenowanym ResNetem, jak twierdził komentarz w `ml/dataset/echo_h5_dataset.py`. Prawdziwy powód:
tak normalizuje Parida. Wartości bez zmian, poprawione samo uzasadnienie — żeby błędne nie trafiło
do rozdziału o implementacji.

---

## 1. Krzywa przy stałym budżecie próbek — **główny rysunek pracy** [Z]

`echo2depth`, **4 próbki na lokalizację (5 496 próbek) w każdym punkcie**, 3 ziarna. Zmienia się
wyłącznie siatka K, z której losowane są kąty. Końcami są istniejące warunki `EA` i `ED`.

| siatka K | RMSE `test@36` | sd (3 ziarna) |
|---|---|---|
| 4 (`cardinal`) | **0,79104** | 0,01066 |
| 6 | 0,70623 | 0,00192 |
| 9 | 0,66342 | 0,00372 |
| 12 | 0,65331 | 0,00490 |
| 18 | 0,64804 | 0,01362 |
| 36 (`random_4`) | **0,64432** | 0,00238 |

`K=4 → K=36`: **−0,13200**, 95 % CI [−0,15178; −0,11231] (bootstrap sparowany po 183 lokalizacjach).

### Punkt odcięcia

| przejście | zysk |
|---|---|
| 4 → 9 | **0,128** |
| 9 → 36 | **0,019** |

Przejście z 4 do 9 orientacji daje **6,7× więcej** niż całe pozostałe rozszerzenie do 36. Krzywa
nasyca się ok. **K = 9–12**.

**To jest wniosek praktyczny, którego nikt wcześniej nie mógł sformułować**, bo nikt nie miał
36 orientacji do podpróbkowania. Przy stałym koszcie generowania (4 rendery na pozycję):
**losować kąty z siatki ~9–12 orientacji zamiast przybijać je do 4 kierunków kardynalnych.**
Rozszerzanie siatki powyżej 12 nie zwraca kosztu.

---

## 2. Pełny model, grupa `glowne` — 1 ziarno [Z-]

Po degradacji z 2026-08-11 §2. `B` policzone wcześniej.

| warunek | RMSE `test@36` |
|---|---|
| `A` `cardinal` | 0,28739 |
| `D` `random_4` | 0,26909 |
| `B` `all` | 0,24205 |

| składowa | wartość |
|---|---|
| gęstość kątowa (D − A) | **−0,01831** |
| ilość danych (B − D) | **−0,02703** |
| łączny (B − A) | −0,04534 |

**[Z-] Zastrzeżenie obowiązkowe:** to jest **n = 1 ziarno**. Obie składowe (0,018 i 0,027) leżą
w przedziale 2,5–12× podłogi szumu frameworka (0,0023–0,0073), ale **bez rozrzutu po ziarnach nie da
się orzec o ich istotności**. Dokładnie to przewidywała degradacja: przewidywany efekt (`bound` =
0,00529) był poniżej progu, więc grupa dostała 1 ziarno dla kompletności protokołu, a nie jako
źródło dowodu.

Znamienne: w pełnym modelu **ilość danych waży więcej niż gęstość** (0,027 wobec 0,018) — odwrotnie
niż w `echo2depth`, gdzie gęstość stanowiła 70,2 %. Zgodne z obrazem, w którym prior wizualny
przykrywa strukturę kątową echa. **Nie należy tego cytować jako wyniku** przy n = 1.

---

## 3. Geometria `patched` na `echo2depth` — wynik **przeciwny do oczekiwanego** [Z-]

| warunek | `patched` | `main` | różnica |
|---|---|---|---|
| `cardinal` | `EPA` 0,79357 | `EA` 0,77895 | **+0,01462** |
| `all` | `EPB` 0,59325 | `EB` 0,58311 | **+0,01014** |
| `random_4` | `EPD` 0,66202 | `ED` 0,64695 | **+0,01507** |

**Domknięcie geometrii pogarsza wynik we wszystkich trzech warunkach**, mimo że dodaje +46,3 %
energii pogłosu (2026-08-10 §2.5).

> **[SKORYGOWANE 2026-08-16 — patrz `RAPORT_SESJI_2026-08-15.md` §3.1.]** Na 3 ziarnach zdanie
> „we wszystkich trzech warunkach" **nie utrzymuje się**: przy `cardinal` Δ zmienia znak
> (−0,00123, p = 0,87) i jest nieodróżnialne od zera. Efekt jest realny wyłącznie w warunkach
> gęstych (`all` +0,01235, p = 0,013; `random_4` +0,01045, p = 0,091) — co jest zgodne
> z mechanizmem opisanym niżej, bo `cardinal` ma najmniej kontrastu kątowego do stracenia.
> Liczby w tabeli powyżej pochodzą z 1 ziarna i **nie należy ich cytować**.

Jest to spójne z drugim pomiarem z tamtej sesji, który wtedy wyglądał na ciekawostkę:
**domknięcie sufitu obniża względny kontrast kątowy pola późnego o 17,5 %** (§2.6). Model uczy się
z **kontrastu między orientacjami**, nie z bezwzględnej energii — więc mocniejszy, ale bardziej
jednorodny pogłos jest gorszym sygnałem. Dwa niezależne pomiary złożyły się w jedno wyjaśnienie.

**[Z-]** n = 1 ziarno; różnice 0,010–0,015 to 1,4–6,5× podłogi szumu. Kierunek jest zgodny we
wszystkich trzech warunkach, co go uwiarygadnia, ale istotności nie orzekamy.

### 3.1 Maska ścisła — kontrola wrażliwości domknięta [Z]

Zastrzeżenie z 2026-08-10 §2.2 mówiło, że piksele „zmienione, a już ważne" (≤ 3,3 % kadru, te które
łata **przesłania**) są nieusuwalną różnicą między wariantami. Teraz da się to sprawdzić — są
checkpointy `patched`. Δ = `patched` − `main` policzone na **trzech** maskach:

| kontrast | maska pełna | **przecięcie** | **ścisła** |
|---|---|---|---|
| `EPA` − `EA` | +0,01462 | +0,00784 | +0,01490 |
| `EPB` − `EB` | +0,01014 | +0,00118 | +0,00446 |
| `EPD` − `ED` | +0,01507 | +0,00861 | +0,01393 |

Ważnych pikseli: przecięcie **89,73 %**, ścisła **88,01 %** kadru.

**Znak jest odporny, wielkość nie.** Wszystkie dziewięć wartości jest dodatnich — `patched` wypada
gorzej niezależnie od maski, więc wniosek z §3 się utrzymuje.

> **[ZAWĘŻONE 2026-08-16.]** Ta tabela jest liczona **na ziarnie 0**. Ponieważ `EPA − EA` zmienia
> znak po dołożeniu ziaren 1–2, zgodność znaku w dziewięciu komórkach jest własnością **tego
> jednego ziarna**, a nie warunku. Zalecenie „raportować Δ na masce przecięcia" **zostaje
> w mocy** — unieważnione jest wyłącznie zdanie o odporności znaku. Ale **wielkość zmienia się 2–8×**
między maskami; największa rozbieżność przecięcie ↔ ścisła to **0,00706**, czyli dokładnie na
górnej granicy podłogi szumu (0,0023–0,0073).

**Werdykt: zastrzeżenia NIE można zdjąć, ale można je ograniczyć liczbą.** Wybór maski przesuwa Δ
o nie więcej niż 0,007 — co przy efektach rzędu 0,01–0,015 jest istotne i musi być w pracy
wymienione. Zalecenie: **raportować Δ na masce przecięcia jako podstawową** (jest konserwatywna —
daje najmniejsze różnice) i podawać maskę ścisłą jako kolumnę odporności.

Uwaga do §3: liczby tam podane są z **maski pełnej**, więc zawyżają Δ wobec przecięcia. Poprawione
wartości podstawowe to **+0,0078 / +0,0012 / +0,0086**, a nie +0,0146 / +0,0101 / +0,0151.
Przy `EPB` różnica jest wtedy praktycznie zerowa.

---

## 4. Model 2, zadanie pretekstowe — działa, ale **z innego powodu, niż zakładaliśmy** [Z]

| wariant | MAAE | top-1 | poziom losowy | błędy sąsiednie |
|---|---|---|---|---|
| K=4 | 61,23° | 47,9 % | 90° / 25 % | 69,4 % |
| K=12 | 55,73° | 27,1 % | 90° / 8,3 % | 39,3 % |
| **K=36** | **25,13°** | 18,3 % | 90° / 2,8 % | 37,8 % |
| K=36 @ 16 par | 61,77° | 5,4 % | 90° / 2,8 % | 12,5 % |

**Zadanie jest wykonalne** — przy K=36 MAAE spada do 25,13° wobec 90° losowego, czyli o 72 %.

### Rozkład efektu — po to był warunek kontrolny

| składowa | wartość |
|---|---|
| **ilość par** (K36 − K36@16par) | **−36,64°** |
| **rozdzielczość kątowa** (K36@16par − K4) | **+0,54°** |

**Cała przewaga K=36 pochodzi z 81× większej liczby par, a rozdzielczość kątowa zadania nie wnosi
nic** (+0,54° to zmiana na niekorzyść, w granicach szumu). Bez tej kontroli wniosek brzmiałby
„gęstsza siatka poprawia zadanie pretekstowe" — i byłby **fałszywy**. To jest ta sama logika, co
warunek `D` w Modelu 1, i drugi raz w tej pracy uratowała przed błędnym wnioskiem.

### Ryzyko z §4.6 się nie zmaterializowało

Przy K=36 MAAE dla przesunięć **≤ 20°** wynosi 26,17°, a dla **> 20°** — 24,98°. Praktycznie tyle
samo, więc **sieć nie jest bezradna przy najdrobniejszej granulacji**. Obawa o SNR ≈ 3,5 przy
rozróżnianiu 0° od 10° była uzasadniona, ale nie potwierdziła się.

Struktura błędów potwierdza, że sieć porządkuje kąty: przy K=36 **37,8 %** błędów trafia w klasę
sąsiednią, przy poziomie losowym **5,7 %** (2 z 35 klas). Przy K=4 wartość 69,4 % jest myląca —
tam poziom losowy to 67 %, więc struktury praktycznie nie ma.

---

## 5. Model 2, transfer — **WYNIK NEGATYWNY** [Z]

Zadanie docelowe: RGB2Depth **bez audio w czasie testu**, 5 ziaren, test Welcha wobec `scratch`.

| inicjalizacja enkodera | RMSE | sd | różnica vs `scratch` | p | odniesienie Gao |
|---|---|---|---|---|---|
| pretrening K=4 | 0,28699 | 0,00433 | −0,00287 | 0,231 | 0,332 |
| pretrening K=36 @ 16 par | 0,28927 | 0,00340 | −0,00059 | 0,751 | — |
| **`Scratch`** | **0,28986** | 0,00204 | — | — | 0,360 |
| pretrening K=36 | 0,29439 | 0,00664 | +0,00453 | 0,207 | — |
| pretrening K=12 | 0,29688 | 0,00657 | +0,00702 | **0,074** | — |

**Pretrening orientacyjny nie poprawia zadania docelowego.** Żadna różnica nie jest istotna.
Najlepszy wariant (K=4) daje **−1,0 %** i mieści się w rozrzucie ziaren; K=12 jest **nominalnie
gorszy** od losowej inicjalizacji na granicy istotności (p = 0,074).

**Porządek warunków też się nie odtwarza.** U Gao poprawa rosła z liczbą klas (2 klasy 0,340 →
4 klasy 0,332); u nas K=36 wypada **gorzej** niż K=4, mimo że rozwiązuje zadanie pretekstowe
nieporównanie lepiej (MAAE 25° wobec 61°).

### Co z tego wynika

Dwie rzeczy, których **nie wolno mylić**:

1. **Zadanie pretekstowe jest rozwiązywalne** (§4) — sieć uczy się przewidywać przesunięcie
   orientacji z pary (widok, echo).
2. **Reprezentacja, której się przy tym uczy, nie pomaga w predykcji głębi z obrazu.** Umiejętność
   nie przenosi się.

To jest wynik negatywny **z kontrolą**, a nie brak wyniku: warunek `K36@16par` pokazuje, że nawet
gdy zadanie pretekstowe jest równie trudne jak K=4, transfer wychodzi tak samo nijako. Efekt nie
zależy więc od trudności zadania.

**Dlaczego tak jest — zdiagnozowane w §5.1.** Trzecia z rozważanych hipotez okazała się trafna:
budżet zadania docelowego wystarcza, żeby losowa inicjalizacja nadgoniła przewagę startową.

### 5.1 DIAGNOZA: dlaczego transfer nie działa [Z]

Cztery pomiary, `outputs/ml/pretext/transfer_diagnosis.json`. Wynik **zmienia interpretację §5**.

**1. Pretrening realnie zmienia koder — nie jest pustą operacją.**
Względna odległość L2 wag enkodera od losowej inicjalizacji: K4 **0,79**, K12 **0,82**,
K36 **0,97**, K36@16par **0,99**. Koder zostaje przebudowany niemal całkowicie.

**2. Zadanie docelowe przepisuje go z powrotem.**
Odległość wag końcowych od tych, którymi zainicjowano: **0,95–0,98**. Czyli 40 000 kroków
RGB2Depth kasuje prawie cały ślad pretreningu.

**3. Końcowe kodery nie pamiętają, skąd startowały.**

| para | odległość |
|---|---|
| `scratch` vs `scratch` (różne ziarna) | **1,256** ← podłoga |
| K4 vs `scratch` | 1,299 |
| K36 vs `scratch` | 1,341 |
| K36@16par vs `scratch` | 1,383 |

Dwa przebiegi z **tej samej** losowej inicjalizacji kończą 1,256 od siebie; pretrenowany od
losowego dzieli 1,30–1,38. **Ten sam rząd wielkości** — inicjalizacja nie zostawia śladu
w rozwiązaniu końcowym.

**4. Ale przewaga startowa JEST — i zanika.**

| krok | `scratch` | K4 | K36 | K36 − `scratch` |
|---|---|---|---|---|
| 1 000 | 0,52803 | 0,49219 | **0,45478** | **−0,07325** |
| 2 000 | 0,45153 | 0,43736 | 0,42524 | −0,02629 |
| 4 000 | 0,39451 | 0,40217 | 0,40737 | +0,01285 |
| 40 000 | 0,31139 | 0,29435 | 0,30062 | −0,01076 |

Na kroku 1 000 pretrenowany K36 jest **o 14 % lepszy** od losowej inicjalizacji (p = 0,060) —
to 10–30× podłoga szumu. Przewaga **znika około kroku 4 000** i dalej fluktuuje wokół zera.

### Wniosek, który zastępuje „transfer nie działa"

> **[WYCOFANE 2026-08-16 — patrz `RAPORT_SESJI_2026-08-15.md` §2.4.]** Przewidywanie postawione
> niżej („przy ograniczeniu zbioru docelowego pretrening powinien zacząć pomagać") zostało
> sprawdzone na 10 % i 25 % zbioru treningowego, 3 ziarna, 18 przebiegów — i **obalone**: przy
> 10 % najlepszy wariant daje Δ = −0,00005 wobec `scratch` (p = 0,99), czyli zero. **Cztery
> pomiary z §5.1 powyżej pozostają w mocy** — fałszywa okazała się wyłącznie interpretacja
> przyczynowa wyciągnięta z nich w tym akapicie. Czytać ten fragment jako hipotezę, która nie
> przeżyła testu, a nie jako wynik.

**Reprezentacja z pretreningu JEST użyteczna. Zadanie docelowe jej nie potrzebuje.**

Zbiór docelowy ma 49 464 próbki i budżet 40 000 kroków — **dość, żeby nauczyć się tych samych cech
od zera**. Pretrening jest więc **nadmiarowy, nie bezużyteczny**: daje realny start, który zostaje
nadgoniony w pierwszych ~10 % budżetu.

To jest zupełnie inne twierdzenie niż „zadanie pretekstowe nic nie wnosi" i **znacznie mocniejsze**,
bo wskazuje warunek, w którym pretrening **musiałby** pomóc: **reżim małej ilości danych**.

**[Z-] Zastrzeżenie do uporządkowania.** Przewaga startowa koreluje z jakością pretreningu
(r = 0,879), ale przy n = 4 wariantach **nie jest to istotne** (p = 0,121), a `K36@16par` psuje
porządek (MAAE 61,77°, a druga co do wielkości przewaga). Uczciwie: **K36 — jedyny wariant, który
rozwiązuje zadanie pretekstowe naprawdę dobrze — ma największą przewagę startową i jako jedyny
zbliża się do istotności.** Pozostałe trzy są między sobą nierozróżnialne.

### Testowalna przewidywalna konsekwencja [X] → **SPRAWDZONA I OBALONA**

Jeśli diagnoza jest trafna, to **przy ograniczeniu zbioru docelowego pretrening powinien zacząć
pomagać**. Eksperyment: powtórzyć transfer na 10 % i 25 % zbioru treningowego (5 000 i 12 400
próbek). Koszt ~1 h GPU, bo krótsze przebiegi. Byłby to najmocniejszy możliwy rozdział o Modelu 2:
nie „nie zadziałało", tylko „działa dokładnie tam, gdzie teoria mówi, że powinno".

> **Wykonane 2026-08-15/16** (`RAPORT_SESJI_2026-08-15.md` §2): 4 946 i 12 366 próbek, podzbiór
> stratyfikowany po lokalizacji, budżet 40 000 kroków bez zmian, 3 ziarna. **Pretrening nie
> zaczyna pomagać** — przy 10 % Δ wobec `scratch` wynosi −0,00005 (K=36) i +0,00439 (K=4),
> obie nieistotne. Koszt był 5,6 h, a nie ~1 h: oszacowanie zakładało krótsze przebiegi, czyli
> skalowany budżet kroków, który został świadomie odrzucony jako przesądzający wynik.

---

## 6. Czego **NIE** zrobiono [X]

- **Nie zdiagnozowano przyczyny negatywnego transferu** (§5) — najważniejsza brakująca rzecz.
- **`glowne` i `geometria_echo` mają n = 1 ziarno**, więc §2 i §3 nie mają orzeczeń o istotności.
- **Grupa `krzywa`** (`C6/C9/C12/C18`, naturalna liczność) nieuruchomiona — odsunięta świadomie,
  bo krzywa stałego budżetu z §1 jest ostrzejsza.
- **Δ na masce przecięcia i ścisłej różni się o 0,007** (§3.1) — na granicy podłogi szumu, więc
  zastrzeżenia o pikselach „zmienionych a ważnych" nie da się całkiem zdjąć, tylko ograniczyć.
- **Model 2 pretrenowany na 1 ziarnie** — MAAE z §4 nie ma rozrzutu.

## 7. Do rozważenia dalej

Trzy kierunki, w kolejności stosunku wartości do kosztu:

1. **Transfer na ograniczonym zbiorze docelowym** (10 % i 25 %, ~1 h) — test przewidywania z §5.1.
   Jeśli pretrening zacznie pomagać, Model 2 dostaje pozytywny wynik z wyjaśnieniem mechanizmu.
2. **Ziarna 1–2 dla `glowne` i `geometria_echo`** (~3,5 h) — dałoby istotność §2 i §3. Wartość
   umiarkowana: bramka już pokazała, że pełny model nie rozdziela efektu gęstości.

Grupa `krzywa` pozostaje odsunięta — krzywa stałego budżetu odpowiada na to samo pytanie ostrzej
i jest już policzona.

---

## 8. Galeria jakościowa — predykcje obok siebie [Z]

`ml/analysis/depth_gallery.py` → `outputs/ml/gallery/depth_gallery.png`. Sześć próbek, **po dwie
z każdej sceny held-out**, wspólna skala 0–14,104 m (przy osobnej normalizacji każdy model
wyglądałby dobrze).

**Dobór próbek: widoki w głąb, nie na ścianę.** Kryterium `p90(głębia) × sd(głębia)` — samo
odchylenie standardowe nie wystarcza, bo ściana pod kątem też je ma wysokie. Mnożenie przez
90. percentyl wybiera kadry z przestrzenią: korytarze i otwarte pomieszczenia.

Kolumny: RGB · prawda · `obraz+echo 36` · `obraz+echo 4` · `SAMO ECHO 36` · `SAMO ECHO 4` ·
`SAM OBRAZ bez echa` · mapa błędu.

RMSE na tych sześciu próbkach (**do podpisu rysunku, nie do tabeli wyników**):

| wersja | RMSE |
|---|---|
| obraz + echo, 36 kątów | **0,4392** |
| obraz + echo, 4 kąty | 0,5054 |
| **sam obraz, bez echa** | **0,5186** |
| SAMO ECHO, 36 kątów | 0,8602 |
| SAMO ECHO, 4 kąty | 1,0901 |

Cztery rzeczy widać na rysunku, których nie widać w liczbach:

1. **Efekt gęstości widać gołym okiem w kolumnach „samo echo".** Wersja 4-kątowa daje rozmytą,
   jednorodną plamę; 36-kątowa odtwarza układ pomieszczenia — jasny obszar w głębi korytarza,
   ciemniejsze ściany po bokach. Różnica 0,230 na tych próbkach.
2. **Samo echo nigdy nie odtwarza obiektów** — rowerów, roślin, krzeseł. Zna geometrię
   pomieszczenia, nie jego zawartość. To granica echa jako źródła geometrii i najlepszy argument
   obrazkowy za łączeniem go z obrazem.
3. **Sam obraz odtwarza kształty, ale myli skalę** — w wierszach `office_4` przewiduje wnękę
   w głębi za daleko. Echo wnosi informację o **bezwzględnej odległości**, której obraz nie ma.
4. **Błąd pełnego modelu koncentruje się na krawędziach i cienkich obiektach** — dokładnie tam,
   gdzie mierzy metryka stratyfikowana.

### [Z-] Uwaga metodologiczna: kolumna „sam obraz" NIE mierzy wkładu echa

Różnica `obraz+echo 36` − `sam obraz` wynosi na tych próbkach **0,079**, ale **nie wolno jej
cytować jako wkładu echa**, bo miesza dwie rzeczy:

| | architektura | parametry |
|---|---|---|
| `obraz + echo` | `RGBDepthNet` + `SimpleAudioDepthNet` + `attentionNet` + `MaterialPropertyNet` | 316 918 781 |
| `sam obraz` | sam `RGBDepthNet` | 16 658 561 |

Te modele różnią się **obecnością echa, mechanizmem uwagi, siecią materiału i 19× liczbą
parametrów naraz**.

**Czystym pomiarem wkładu echa pozostaje warunek `SE`** (2026-08-11 §2): ta sama architektura, ten
sam budżet, permutowane wyłącznie przypisanie echa do obrazu. Daje **0,0223** na pełnym zbiorze
testowym [+0,01840; +0,02643].

Czyli z 0,079 obserwowanych na tych próbkach **~0,022 to echo, a reszta to pojemność i architektura**.
Kolumna „sam obraz" jest cenna **wizualnie** — pokazuje, co obraz potrafi sam — ale liczba pod nią
nie jest odpowiedzią na pytanie „ile dodaje echo".

Kolumnę `SE` usunięto z domyślnego zestawu na życzenie autora; wraca przez
`--runs ... SE_seed0`.
