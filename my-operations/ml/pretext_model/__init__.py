"""MODEL 2 -- zadanie pretekstowe przewidywania WZGLEDNEJ ORIENTACJI.

To jest miejsce, w ktorym 36 orientacji faktycznie sie oplaca, i to nie jest
dodatek do Modelu 1 -- to jest teza z postera (zmusic koder wizualny do
glebszego rozumienia geometrii).

Trzy powody, dla ktorych ma pierwszenstwo przed rozbudowa Modelu 1, wszystkie
sprawdzalne w zrodlach:

1. GAO SAM ZROBIL ABLACJE PO LICZBIE KLAS I ZATRZYMAL SIE NA 4 Z POWODU
   NARZEDZIA. Tabela 3 pracy glownej: `Scratch` 0,360 -> `SimpleVisualEchoes`
   (2 klasy) 0,340 -> `VisualEchoes` (4 klasy) 0,332. Trend monotoniczny.
   Rozszerzenie na 12 i 36 klas jest przedluzeniem ICH WLASNEJ OSI, a nie nowym
   pomyslem do obrony.

2. ZADANIE ROSNIE KWADRATOWO, NIE LINIOWO. Para to (widok z orientacji i, echo
   z orientacji j) z tej samej lokalizacji, a etykieta to przesuniecie (j - i):

       K   par na lokalizacje   par treningowych (1 374 lokalizacje)
       4          16                    21 984
      12         144                   197 856
      36       1 296                 1 780 704

   81x wiecej sygnalu uczacego z TYCH SAMYCH renderow, wobec 9x w Modelu 1.

3. Bez tego wstep pracy trzeba przepisac pod inna hipoteze niz zgloszona.

Specyfikacja trzymana doslownie za suplementem Gao (§I) -- patrz `model.py`.
Podzial: ten sam, co w Modelu 1 (`outputs/ml/splits/replica_locations.json`,
odcisk `e0bf7547668d9e0a`), bo Gao rowniez trenuje zadanie docelowe na tych
samych 15 scenach, na ktorych robil pretrening.
"""

from __future__ import annotations

# Warianty liczby klas. K musi rownomiernie dzielic 36, bo klasy powstaja z
# roznic katow na podsiatce K orientacji -- inaczej przesuniecie (j-i) nie
# wpadaloby dokladnie w zadna klase.
K_VARIANTS = (4, 12, 36)

# Kontrola przy rownej liczbie par (4.4): K=36 podprobkowane do 16 par na
# lokalizacje, czyli dokladnie tyle, ile ma K=4.
PAIRS_PER_LOCATION_CONTROL = 16
