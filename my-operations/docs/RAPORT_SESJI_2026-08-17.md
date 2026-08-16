# Raport sesji — 2026-08-17: diagnoza negatywnego transferu. **Zamknięcie fazy eksperymentalnej**

Jedno pytanie: **dlaczego pretrening orientacyjny nie pomaga w predykcji głębi**, mimo że samo
zadanie pretekstowe jest rozwiązane bardzo dobrze. Odpowiedź szukana przez **sondowanie
zamrożonych reprezentacji** — żaden enkoder nie był w tej sesji uczony.

Kolejka: **22 kroki, 2 h 13 min, 0 błędów.** Wszystkie 21 przebiegów sond z **potwierdzonym
sumą kontrolną zamrożeniem enkodera**. Plik dowodowy:
`outputs/ml/echo_ablation/final_results_2026-08-15.json` (klucze `sonda_glebi`,
`sondy_pomocnicze`, `cka`). Reguła interpretacyjna zapisana przed pomiarem:
`outputs/ml/probing/REGULA_PRZED_POMIAREM.md`.

Statusy: **[Z]** zmierzone · **[Z-]** z zastrzeżeniem · **[W]** wywnioskowane · **[X]** niesprawdzone.

---

## 0. Odpowiedź na pytanie sesji [Z]

**Reprezentacja z pretreningu orientacyjnego zawiera dużo informacji o głębi — niepowodzenie
transferu nie leży w treści reprezentacji, tylko w tym, co robi z nią dostrajanie.**

Zamrożony enkoder pretekstowy `K=36` pozwala odczytać głębię o **0,16787 RMSE lepiej** niż
zamrożony enkoder **losowy** (p = 0,0004; **23× podłoga szumu**). To pokrywa **63,4 %** całej
rozpiętości między enkoderem losowym a enkoderem, który przeszedł pełne uczenie na głębi.
Jakość tej reprezentacji **rośnie z jakością pretreningu**: `K=36` (MAAE 25,65°) daje 63,4 %,
`K=4` (MAAE 59,94°) — 33,1 %.

Jednocześnie standardowe dostrajanie całej sieci nie daje z tego **nic** (wszystkie p > 0,07,
także przy 10 % zbioru docelowego). Zestawione z pomiarami z 2026-08-13 §5.1, które pozostają
w mocy — dostrajanie przepisuje enkoder niemal całkowicie (odległość wag końcowych od startowych
0,95–0,98), a końcowe enkodery nie pamiętają, skąd startowały — daje to spójny obraz:

> **Zadanie docelowe nie buduje na pretrenowanej reprezentacji, tylko ją niszczy.**
> Cechy są dobre; optymalizacja je wyrzuca.

To jest **inne twierdzenie niż „pretrening nic nie wnosi"** i znacznie mocniejsze, bo wskazuje
konkretne miejsce awarii — protokół dostrajania, a nie zadanie pretekstowe.

**Drugie ustalenie, niezależne od pierwszego** (kontrola `K36@16par`, §1): przewaga `K=36` nad
`K=4` rozkłada się prawie po równo na **gęstość kątową danych** (+14,0 pp, p = 0,0042) i **liczbę
par** (+16,3 pp, p = 0,0001). Gęstość kątowa poprawia zawartość geometryczną kodera **mimo że
w samym zadaniu pretekstowym nie poprawiała wyniku w ogóle** (−1,24°, p = 0,83 w 2026-08-13 §4).
Jest to **dysocjacja**: MAAE zadania pretekstowego nie jest wskaźnikiem tego, czego uczy się koder.

---

## 1. Sonda głębi — rdzeń diagnozy [Z]

Enkoder zamrożony (`requires_grad=False` **oraz** `eval()`), uczony wyłącznie dekoder. Budżet,
strata, walidacja i test bez zmian wobec `transfer.py`. 3 ziarna.

| enkoder (zamrożony) | RMSE sondy | sd | Δ vs `random` | p | pokrycie rozpiętości |
|---|---|---|---|---|---|
| `depth_trained` *(górna granica)* | **0,29234** | 0,00157 | −0,26462 | 0,0002 | 100 % |
| **`pretext_K36`** | **0,38908** | 0,00243 | **−0,16787** | **0,0004** | **63,4 %** |
| `pretext_K36@16par` *(kontrola)* | 0,43232 | 0,00154 | −0,12463 | 0,0012 | **47,1 %** |
| `pretext_K4` | 0,46932 | 0,00535 | −0,08763 | 0,0004 | 33,1 % |
| `random` *(podłoga)* | 0,55695 | 0,00878 | — | — | 0 % |

Oba kryteria z reguły przedrejestrowanej spełnione łącznie: `p < 0,05` **oraz** różnica ponad
podłogą szumu (0,0073). Δ dla `K=36` jest **23×** większa od podłogi.

**Werdykt wg reguły zapisanej przed pomiarem: gałąź druga — „cechy SĄ użyteczne, problem leży
w dynamice dostrajania".**

**Rysunek:** `outputs/ml/figures/rys_4_sonda_glebi.png` (podpis i uzasadnienia — `docs/RYSUNKI.md`).

### Dlaczego kontrola z losowym enkoderem była niezbędna

`random` daje RMSE **0,55695**, a nie wartość bliską bezużyteczności. To potwierdza zastrzeżenie
zapisane z góry: `RGBDepthNet` jest U-Netem z **połączeniami skrótowymi**, więc
`rgbdepth_conv1feature` (64 kanały w pełnej rozdzielczości) trafia wprost do ostatniej warstwy
dekodera — nawet losowy enkoder podaje użyteczne krawędzie. Gdyby raportować samo `S_K36 = 0,389`
bez tej podłogi, liczba nie miałaby żadnej skali. **Wielkością rozstrzygającą jest różnica, nie
wartość bezwzględna.**

### Monotoniczność względem jakości pretreningu [Z]

| wariant pretreningu | MAAE zadania pretekstowego | pokrycie rozpiętości głębi |
|---|---|---|
| `K=36` | 25,65° | **63,4 %** |
| `K=4` | 59,94° | 33,1 % |
| brak (losowy) | 90° (poziom losowy) | 0 % |

**Im lepiej rozwiązane zadanie orientacyjne, tym więcej informacji o głębi w zamrożonym
enkoderze.** To jest bezpośrednie, ilościowe poparcie tezy z postera („zagęszczenie ech zmusza
koder wizualny do głębszego rozumienia geometrii") — teza zostaje **potwierdzona**, a nie
sfalsyfikowana, i to na wielkości mierzonej niezależnie od zadania docelowego.

### Rozkład przewagi `K=36` nad `K=4` — kontrola `K36@16par` [Z]

Przewaga `K=36` nad `K=4` miała dwa możliwe źródła: **jakość pretreningu** albo **81× większą
liczbę par**. Warunek `K36@16par` (gęsta siatka 36 orientacji, budżet par równy `K=4`) je
rozdziela. Reguła zapisana przed pomiarem przewidywała trzy rozłączne wyniki; wypadł **trzeci,
pośredni**:

| składowa | porównanie | ΔRMSE | pp rozpiętości | p |
|---|---|---|---|---|
| **gęstość kątowa danych** | `K36@16par` − `K4` *(ten sam budżet 16 par, siatka 36 vs 4)* | −0,03700 | **+14,0** | **0,0042** |
| **liczba par** | `K36` − `K36@16par` *(ta sama siatka 36, 1 296 vs 16 par)* | −0,04324 | **+16,3** | **0,0001** |

**Oba czynniki wnoszą, prawie po równo, i oba są istotne.**

#### To jest DYSOCJACJA wobec rozkładu samego zadania pretekstowego [Z]

| rozkład | gęstość / rozdzielczość kątowa | liczba par |
|---|---|---|
| **MAAE zadania pretekstowego** (2026-08-13 §4) | −1,24°, **p = 0,83** (nic) | −33,05°, p = 0,015 |
| **sonda głębi** (ta sesja) | **+14,0 pp, p = 0,0042** | +16,3 pp, p = 0,0001 |

W samym zadaniu pretekstowym gęstsza siatka **nie poprawiała wyniku** — cała przewaga pochodziła
z liczby par. W zamrożonym koderze wizualnym gęstsza siatka **poprawia zawartość geometryczną
o 14 punktów procentowych rozpiętości**, mimo że `K36@16par` ma MAAE 58,70°, czyli praktycznie
tyle samo co `K4` (59,94°).

> **Gęstsze echo kształtuje koder wizualny nawet wtedy, gdy nie poprawia wyniku w samym zadaniu
> orientacyjnym.** MAAE zadania pretekstowego **nie jest** dobrym wskaźnikiem tego, czego uczy się
> koder — dwie sieci równie słabe w zadaniu (58,70° i 59,94°) różnią się o 14 pp w zawartości
> informacji o głębi.

Dla tezy z postera to jest **wzmocnienie, nie osłabienie**: zagęszczenie ech działa na koder
**bezpośrednio**, przez różnorodność sygnału treningowego, a nie pośrednio przez poprawę wyniku
w zadaniu pretekstowym. Tego nie dałoby się zobaczyć ani z samego MAAE, ani z samego transferu.

---

## 2. Sondy pomocnicze — hipoteza ze zlecenia **OBALONA** [Z]

Liniowa głowa na uśrednionych przestrzennie cechach `conv5`, te same zamrożone enkodery, 3 ziarna.

| zadanie | poziom losowy | `pretext_K36` | `random` | `depth_trained` | Δ K36−random | p |
|---|---|---|---|---|---|---|
| orientacja bezwzględna (MAAE) | 90,0° | 64,92 ± 0,11 | 70,84 ± 0,38 | **59,61 ± 0,45** | −5,92 | 0,0006 |
| tożsamość sceny (top-1) | 6,7 % | 64,0 ± 0,2 % | 60,9 ± 0,3 % | **74,4 ± 0,2 %** | +3,1 pp | 0,0006 |

Hipoteza ze zlecenia brzmiała: enkoder pretekstowy powinien być **wyraźnie lepszy** od losowego
przy orientacji i tożsamości sceny, a **nierozróżnialny** przy głębi — co opisywałoby zadanie
pretekstowe jako wymuszające płytką dyskryminację widoku zamiast geometrii metrycznej.

**Wyszło dokładnie odwrotnie.** Przy głębi `K=36` bije losowy o 63,4 % rozpiętości; przy
orientacji i tożsamości sceny przewaga jest **niewielka** (5,9° z 90; 3,1 punktu procentowego),
a w obu tych zadaniach **najlepszy jest `depth_trained`** — enkoder, który nigdy nie widział
zadania orientacyjnego.

Wniosek: cechy uczone przez zadanie pretekstowe **nie są cechami „rozpoznaj, w którą stronę
patrzysz"**. Są cechami geometrycznymi, użytecznymi dla głębi — a orientacja i tożsamość sceny
wychodzą z nich mimochodem, słabiej niż z cech uczonych wprost na głębi. Hipoteza o płytkiej
dyskryminacji widoku jest **obalona**; teza z postera **utrzymana**.

---

## 3. CKA — podobieństwo warstwa po warstwie [Z]

Liniowe CKA na cechach uśrednionych przestrzennie, 1 000 obrazów walidacyjnych.

| para | conv1 | conv2 | conv3 | conv4 | conv5 |
|---|---|---|---|---|---|
| `pretext_K36` ↔ `depth_trained` | 0,996 | 0,982 | 0,953 | 0,964 | **0,943** |
| `random` ↔ `depth_trained` *(podłoga)* | 0,986 | 0,925 | 0,888 | 0,905 | **0,902** |
| `pretext_K36` ↔ `random` | 0,974 | 0,958 | 0,925 | 0,885 | 0,906 |

Enkoder pretekstowy jest **bliżej enkodera głębi niż enkoder losowy na każdej warstwie**, a
przewaga **rośnie z głębokością**: na `conv1` różnica wobec podłogi wynosi 0,010, na `conv5` już
0,041. Czyli podobieństwo nie bierze się z dzielenia cech niskopoziomowych — narasta dokładnie
tam, gdzie zaczyna się specyfika zadania. Spójne z §1.

**[Z-] Zastrzeżenie.** CKA liczone na cechach **uśrednionych przestrzennie** gubi strukturę
przestrzenną, dlatego wszystkie wartości są wysokie (0,88–1,00) i liczy się wyłącznie ich
**uporządkowanie względem podłogi**, nie wartości bezwzględne.

---

## 4. Co to zmienia w rozdziale o Modelu 2

Rozdział przestaje kończyć się pytaniem. Nowa struktura wniosku:

1. **Zadanie pretekstowe jest rozwiązywalne** — MAAE 25,65 ± 0,74° wobec 90° losowego.
2. **Uczy koder wizualny cech geometrycznych** — 63,4 % rozpiętości do enkodera uczonego wprost
   na głębi, monotonicznie z jakością pretreningu.
3. **Standardowe dostrajanie tego nie wykorzystuje** — zero istotnej poprawy, także przy 10 %
   zbioru; dostrajanie przepisuje enkoder (odległość 0,95–0,98) i zaciera ślad inicjalizacji.
4. **Wąskim gardłem jest protokół przenoszenia, nie pretrening.**

To jest wynik **pozytywny z jasno wskazanym ograniczeniem**, a nie negatywny bez wyjaśnienia.
Poprzednia formuła „pretrening orientacyjny nie przenosi się na predykcję głębi" jest zbyt
mocna — poprawna wersja brzmi: **„reprezentacja przenosi się, standardowe dostrajanie jej nie
zachowuje"**.

**Bezpośrednia konsekwencja praktyczna [W]:** protokoły, które chronią pretrenowane cechy —
zamrożenie enkodera, mniejszy krok uczenia na enkoderze, stopniowe odmrażanie — są tu
**przewidywalnie skuteczne**, bo sonda pokazuje, że jest co chronić. Nie zostało to zmierzone
i jest kandydatem numer jeden do rozdziału o dalszych badaniach.

---

## 5. Czego **NIE** zrobiono [X]

- **Nie zmierzono protokołów chroniących cechy** (zamrożenie, niższy `lr` na enkoderze,
  stopniowe odmrażanie) — to jest przewidywanie z §4, nie wynik.
- **Sondy pomocnicze na cechach uśrednionych przestrzennie** — sonda z zachowaną strukturą
  przestrzenną (np. na pełnej mapie `conv5`) mogłaby dać inny obraz orientacji.
- **CKA tylko na cechach uśrednionych** — patrz zastrzeżenie w §3.
- Bez zmian: grupy `krzywa` i `geometria` skreślone, `ESA` niezaimplementowane, mp3d niepoliczone.

---

## 6. Twierdzenia oparte na 1 ziarnie — do cytowania z zastrzeżeniem

Ustalenie z 2026-08-15 §0 brzmiało: **każde** twierdzenie przeliczone z 1 na 3 ziarna wymagało
korekty. Poniższe pozostają na jednym ziarnie i w tekście muszą nieść to zastrzeżenie:

| twierdzenie | wartość | dlaczego 1 ziarno |
|---|---|---|
| `c_full` — całkowity wkład echa w pełnym modelu | 0,02228 [0,01840; 0,02643] | bramka, nie pozycja w tabeli wyników; CI z bootstrapu po lokalizacjach |
| Analiza masek — udział pikseli „zmienionych, a ważnych" | 3,3 % kadru | pomiar właściwości zbioru, nie modelu |
| Rozkład MAAE ≤ 20° / > 20° w zadaniu pretekstowym | 26,17° / 24,98° | policzone tylko na ziarnie 0 |

Enkodery użyte w sondach (`pretext_K36`, `pretext_K4`, `depth_trained`) też są **pojedynczymi
checkpointami** — 3 ziarna sondy mierzą rozrzut **dekodera**, nie rozrzut pretreningu. Wielkość
`S_K36 − S_rand` = 0,168 jest jednak 23× ponad podłogą szumu, więc rozrzut enkodera musiałby być
ekstremalny, żeby to odwrócić.

---

## 7. Faza eksperymentalna — **ZAMKNIĘTA**

**Tabela zamykająca — każde twierdzenie, które trafia do pracy.** Do przepisywania liczb służy
`docs/STAN_WYNIKOW.md`; ta tabela jest mapą: co jest ustalone, na ilu ziarnach i skąd.

| # | twierdzenie | liczba | ziaren | status | plik dowodowy |
|---|---|---|---|---|---|
| 1 | Gęstość kątowa poprawia `echo2depth` | D−A = 0,14672 ± 0,01303 | **3** | [Z] | `echo_ablation/echo_3seeds.json` |
| 2 | **Krzywa nasyca się przy K = 9–12** | 4→9: 0,128 · 9→36: 0,019 | **3** | [Z] | `final_results_2026-08-13.json` |
| 3 | Gęstość > ilość danych w `echo2depth` | udział 70,2 ± 3,0 % | **3** | [Z] | `echo_ablation/echo_3seeds.json` |
| 4 | Efekt utrzymuje się w pełnym modelu | D−A = 0,02048 ± 0,00350, p = 0,0096 | **3** | [Z] | `final_results_2026-08-15.json` |
| 5 | Pełny efekt 4→36 w pełnym modelu | B−A = 0,04881, **−16,7 %**, p = 0,0018 | **3** | [Z] | `final_results_2026-08-15.json` |
| 6 | W pełnym modelu ilość danych > gęstość | udział gęstości 42,0 % | **3** | [Z] | `final_results_2026-08-15.json` |
| 7 | Kara baseline'u powstaje na kątach niewidzianych | 91,4 % kary `EA` | **3** | [Z] | `echo_ablation/echo_3seeds.json` |
| 8 | Efekt gęstości **nie zależy od geometrii** | main −0,14672 vs patched −0,13504, p = 0,26 | **3** | [Z] | `final_results_2026-08-15.json` |
| 9 | Domknięcie geometrii szkodzi tylko przy gęstej siatce | `all` +0,01235 (p = 0,017), maska pełna | **3** | [Z-] | `final_results_2026-08-15.json` |
| 10 | Model rzeczywiście używa echa (bramka) | c_full = 0,02228 [0,0184; 0,0264] | **1** | [Z-] | `echo_ablation/full_model_gate.json` |
| 11 | Zadanie pretekstowe jest rozwiązywalne | MAAE 25,65 ± 0,74° wobec 90° | **3** | [Z] | `pretext/summary.json` |
| 12 | W zadaniu pretekstowym liczy się liczba par, nie rozdzielczość | −33,05° (p = 0,015) vs −1,24° (p = 0,83) | **3** | [Z] | `pretext/summary.json` |
| 13 | Pretrening **nie** poprawia zadania docelowego | wszystkie p > 0,07 | **5** | [Z] | `final_results_2026-08-13.json` |
| 14 | Ograniczenie zbioru docelowego **nie** pomaga | @10 %: Δ = −0,00005, p = 0,99 | **3** | [Z] | `final_results_2026-08-15.json` |
| 15 | **Cechy pretreningu NIOSĄ informację o głębi** | −0,16787 vs losowy (63,4 % rozpiętości), p = 0,0004 | **3** | [Z] | `final_results_2026-08-15.json` |
| 16 | Gęstość i liczba par wnoszą po równo do kodera | +14,0 pp (p = 0,0042) · +16,3 pp (p = 0,0001) | **3** | [Z] | `final_results_2026-08-15.json` |
| 17 | **Dysocjacja: MAAE nie jest wskaźnikiem jakości kodera** | gęstość: p = 0,83 w MAAE, p = 0,0042 w sondzie | **3** | [Z] | jw. + `pretext/summary.json` |
| 18 | To nie są cechy „rozpoznaj orientację" | przy orientacji i scenie najlepszy `depth_trained` | **3** | [Z] | `final_results_2026-08-15.json` |
| 19 | Koder pretekstowy bliżej kodera głębi niż losowy | CKA conv5: 0,943 vs 0,902 (podłoga) | 1 | [Z-] | `probing/cka.json` |

Twierdzenia **2** i **15–17** są oryginalnym wkładem pracy — żadnego nie dało się sformułować
bez 36 orientacji do podpróbkowania i bez sondowania zamrożonych reprezentacji.

Łącznie w fazie ML: **84 przebiegi GPU**, ostatnie cztery kolejki bez błędu.
**Do policzenia nie zostało nic.**
