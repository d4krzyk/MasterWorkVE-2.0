# Rysunki do pracy — podpisy i uzasadnienia

Wszystkie rysunki powstają z **zapisanych plików dowodowych**, nie z żywych przebiegów:

```
python my-operations/ml/analysis/figures.py        # rys. 1–3, zero GPU, ~5 s
python my-operations/ml/analysis/depth_gallery.py  # rys. 4 (galeria), wymaga GPU
```

Wyjście: `outputs/ml/figures/` i `outputs/ml/gallery/`.

**Konwencja wspólna dla rysunków 1–3.** Paleta kategoryczna zwalidowana pod kątem rozróżnialności
przy zaburzeniach widzenia barw (najgorsza para: ΔE 9,2 w deuteranopii, 27,6 w widzeniu
normalnym — progi to odpowiednio 8 i 15). Każda seria jest **opisana bezpośrednio przy krzywej**,
nie tylko w legendzie, żeby rysunek pozostał czytelny po wydruku w skali szarości. Wąsy to
**rzeczywiste odchylenie standardowe po ziarnach**, nie przedział ufności ani błąd standardowy.

---

## Rysunek 1 — krzywa nasycenia (`rys_1_krzywa_nasycenia.png`)

> **Rys. 1.** Nasycenie efektu gęstości kątowej przy stałym budżecie próbek. Sieć głębi (tylko
> echo), RMSE na zbiorze testowym obejmującym wszystkie 36 orientacji. W każdym punkcie krzywej
> zbiór treningowy liczy **dokładnie 5 496 próbek** (4 na lokalizację); zmienia się wyłącznie
> siatka *K*, z której losowane są kąty. Punkty to średnie z 3 ziaren, wąsy to odchylenie
> standardowe. Przejście z 4 do 9 orientacji daje **−0,128 RMSE**, a całe pozostałe rozszerzenie
> z 9 do 36 tylko **−0,019** — 6,7× mniej. Krzywa nasyca się w okolicy *K* = 9–12 (obszar
> zacieniony).

**Co ten rysunek dowodzi, a czego nie.** Dowodzi, że przy **stałym koszcie generowania danych**
opłaca się losować kąty z gęstszej siatki, a nie przybijać je do 4 kierunków kardynalnych — i że
korzyść wyczerpuje się przy ~9–12 orientacjach. Nie dowodzi, że więcej danych nie pomaga: oś
liczności jest tu **celowo zamrożona**, a jej wpływ mierzy osobno rysunek 3.

**Dlaczego oś X jest logarytmiczna.** *K* to liczebność siatki, wielkość w skali ilorazowej.
Na osi liniowej wartości 4–12 — czyli cały zakres, w którym leży wniosek pracy — ścisnęłyby się
przy lewej krawędzi, a połowę szerokości rysunku zająłby nieinformatywny odcinek 18–36.

**Zastrzeżenie do zacytowania razem z rysunkiem.** Przy *K* = 18 odchylenie standardowe wynosi
0,0136 RMSE, czyli **więcej niż cała różnica między *K* = 18 a *K* = 36**. Ogon krzywej niczego
nie rozstrzyga i nie należy go interpretować; rozstrzygający jest odcinek 4 → 9.

---

## Rysunek 2 — generalizacja kątowa (`rys_2_generalizacja_katowa.png`)

> **Rys. 2.** Błąd predykcji w funkcji odległości testowanego kąta od najbliższego kąta obecnego
> w zbiorze treningowym. Sieć głębi (tylko echo), 3 ziarna, wąsy to odchylenie standardowe.
> Warunek `EA` (4 kierunki kardynalne, baseline VisualEchoes) pogarsza się monotonicznie:
> od 0,596 przy kątach widzianych do 0,911 przy odchyleniu 40°, czyli o **53 %**. Linie poziome
> to warunki `ED` (4 kąty losowane z 36) i `EB` (wszystkie 36), które takiej osi nie mają.

**Dlaczego `ED` i `EB` są liniami, a nie krzywymi.** To nie jest brak danych. Oba warunki uczą się
na kątach rozłożonych po całej siatce, więc **każdy** kąt testowy leży dla nich w odległości 0 od
zbioru treningowego — oś „odległość od siatki" ma dla nich jeden punkt. Narysowanie ich jako
krzywych sugerowałoby pomiar, którego wykonać się nie da.

**Co ten rysunek wyjaśnia.** Rozkłada karę baseline'u 4-kierunkowego na dwie składowe: 91,4 %
powstaje na orientacjach, których model **nie widział**, a nie na gorszym nauczeniu tych, które
widział. Baseline Gao głównie **nie pokrywa przestrzeni orientacji** — to jest inna wada niż
„uczy się gorzej" i prowadzi do innego zalecenia projektowego.

**Zastrzeżenie.** Na kątach, które `EA` widział, różnica wobec `EB` jest niezerowa (+0,01787 ±
0,01128, ~2,4× podłogi szumu). **Nie wolno pisać, że modele są tam identyczne** — wcześniejsza
wersja tego twierdzenia była artefaktem doboru punktu kontrolnego.

---

## Rysunek 3 — rozkład efektu (`rys_3_rozklad_efektu.png`)

> **Rys. 3.** Rozkład poprawy RMSE na dwie składowe — samą gęstość kątową (różnica `D − A`, przy
> równej liczbie próbek) i samą ilość danych (różnica `B − D`, przy równej gęstości) — dla obu
> architektur. 3 ziarna, wąsy to odchylenie standardowe. W sieci opartej wyłącznie na echu gęstość
> kątowa odpowiada za **70,2 %** efektu łącznego; w sieci pełnej, gdzie obraz RGB niesie większość
> informacji o głębi, jej udział spada do **42 %**, a sam efekt maleje 7,2× (0,147 → 0,020).
> **Uwaga: panele mają różne skale osi Y.**

**Dlaczego dwa panele, a nie jeden wykres.** Efekty różnią się 7,2×. Na wspólnej osi słupki
pełnego modelu byłyby praktycznie niewidoczne, a wykres z dwiema osiami Y jest w tym miejscu
najgorszym możliwym wyborem — pozwala dowolnie ustawić względne wysokości słupków i przez to
zasugerować dowolny wniosek. Dwa panele z jawnie oznaczoną różnicą skal mówią to samo, nie dając
się tak nagiąć.

**Warunek `D` jest tu kluczowy.** Ma **dokładnie tyle samo próbek** co `A` (5 496), ale kąty
losowane są per lokalizacja, więc model widzi wszystkie 36 orientacji — tylko nie wszystkie
z każdego punktu. Bez niego różnica `B − A` mieszałaby gęstość z rozmiarem zbioru i żadnej z nich
nie dałoby się orzec.

**Co pokazuje spadek udziału z 70 % do 42 %.** Prior wizualny przykrywa część struktury kątowej
echa — to jest oczekiwane i **nie unieważnia** wniosku: w pełnym modelu efekt gęstości pozostaje
istotny (p = 0,0096, przedział bootstrapowy wyklucza zero w 3 ziarnach na 3).

---

## Rysunek 4 — galeria predykcji (`gallery/depth_gallery.png`)

> **Rys. 4.** Predykcje głębi dla sześciu próbek ze scen wyłączonych z treningu, po dwie z każdej
> sceny. Kolumny: obraz RGB · prawda · obraz + echo (36 orientacji) · obraz + echo (4) · samo echo
> (36) · samo echo (4) · sam obraz · mapa błędu. Wspólna skala 0–14,104 m we wszystkich kolumnach.

**Dobór próbek.** Kryterium `p90(głębia) × sd(głębia)` — samo odchylenie standardowe nie
wystarcza, bo ściana pod kątem też ma je wysokie. Mnożenie przez 90. percentyl wybiera kadry
z przestrzenią: korytarze i otwarte pomieszczenia, gdzie echo ma co mierzyć.

**Wspólna skala jest istotna.** Przy osobnej normalizacji każdy model wygląda dobrze.

**[!] Kolumna „sam obraz" NIE mierzy wkładu echa.** Różnica wobec „obraz + echo" wynosi na tych
próbkach 0,079, ale miesza **cztery** rzeczy naraz: obecność echa, mechanizm uwagi, sieć materiału
i 19× większą liczbę parametrów (316,9 M wobec 16,7 M). Czystym pomiarem wkładu echa jest warunek
`SE` — ta sama architektura i budżet, permutowane wyłącznie przypisanie echa do obrazu — który
daje **0,0223** [+0,01840; +0,02643]. Z tych 0,079 około 0,022 to echo, reszta to pojemność
i architektura. Kolumna jest cenna wizualnie; liczba pod nią nie odpowiada na pytanie „ile dodaje
echo".

**RMSE na tych sześciu próbkach — do podpisu rysunku, nie do tabeli wyników:** obraz + echo 36:
0,4392 · obraz + echo 4: 0,5054 · sam obraz: 0,5186 · samo echo 36: 0,8602 · samo echo 4: 1,0901.
