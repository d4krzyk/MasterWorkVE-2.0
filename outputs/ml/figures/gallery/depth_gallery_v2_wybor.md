# Galeria v2: wybrane próbki

Źródło reguły: tabela per probka eval/B_seed0/samples_test@36.npz. RMSE próbki = sqrt(Σ błędu² / liczba pikseli z prawdą > 0), model B, ziarno 0, test@36.

**Reguła (nie ręczna):** w każdej z trzech scen testowych wybrano próbkę, której RMSE jest najbliższe medianie RMSE sceny („typowy”) oraz 90. percentylowi („trudny”); remis rozstrzyga kolejność w tabeli. Percentyl: interpolacja liniowa (`numpy.percentile`).

**Kontrola:** RMSE wybranych próbek policzone z modelu B (`best_*.pth`, AMP) musi zgadzać się z tabelą per próbka z tolerancją 0.001.

| scena | lokalizacja | kąt [°] | rodzaj | cel (percentyl sceny) [m] | RMSE B, tabela [m] | RMSE B, model [m] | \|różnica\| | kontrola |
|---|---|---|---|---|---|---|---|---|
| apartment_2 | 81 | 320 | typowy (mediana) | 0.23670 | 0.23670 | 0.23672 | 2.2e-05 | OK |
| apartment_2 | 83 | 350 | trudny (90. percentyl) | 0.40867 | 0.40887 | 0.40888 | 1.1e-05 | OK |
| frl_apartment_5 | 82 | 20 | typowy (mediana) | 0.16793 | 0.16807 | 0.16808 | 5.5e-06 | OK |
| frl_apartment_5 | 55 | 310 | trudny (90. percentyl) | 0.38947 | 0.38961 | 0.38959 | 2.1e-05 | OK |
| office_4 | 28 | 30 | typowy (mediana) | 0.17171 | 0.17184 | 0.17185 | 8.5e-06 | OK |
| office_4 | 54 | 200 | trudny (90. percentyl) | 0.29806 | 0.29786 | 0.29784 | 1.4e-05 | OK |

Wynik kontroli: wszystkie 6 zgodne (tolerancja 0.001).
