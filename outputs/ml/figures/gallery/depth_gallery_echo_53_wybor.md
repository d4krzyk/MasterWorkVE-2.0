# Galeria echo 5.3: wybrane próbki

Źródło wyboru: tabela per próbka eval/EB_seed0/samples_test@36.npz. RMSE próbki = sqrt(Σ błędu² / liczba pikseli z prawdą > 0), test@36, ziarno 0.

**Reguła (nie ręczna):** w każdej z trzech scen testowych wybrano próbkę o RMSE EB najbliższym medianie sceny („typowy”, wersja główna) oraz najbliższym 90. percentylowi („trudny”, wersja dodatkowa razem z typowymi). Percentyl: interpolacja liniowa; remis rozstrzyga kolejność w tabeli.

**Kontrola:** RMSE policzone z modeli (best_*.pth, AMP, prawdziwe echo także dla ESE) vs tabela per próbka danego modelu, tolerancja 0.001.

Odległość = odległość kąta od najbliższej orientacji kardynalnej (0/90/180/270°), czyli odległość dla EA.

| scena | lok. | kąt [°] | rodzaj | odl. od kard. [°] | RMSE EA | RMSE ED | RMSE EB | RMSE ESE | max \|tab − model\| |
|---|---|---|---|---|---|---|---|---|---|
| apartment_2 | 153 | 230 | typowy | 40 | 0.5592 | 0.4713 | 0.4986 | 0.6024 | 1.0e-04 |
| apartment_2 | 90 | 280 | trudny | 10 | 0.9268 | 0.8443 | 1.1517 | 0.8981 | 2.7e-04 |
| frl_apartment_5 | 93 | 170 | typowy | 10 | 0.4098 | 0.5937 | 0.3410 | 1.1499 | 1.8e-04 |
| frl_apartment_5 | 80 | 240 | trudny | 30 | 1.2024 | 0.7890 | 0.7917 | 2.1267 | 1.1e-04 |
| office_4 | 66 | 300 | typowy | 30 | 0.4758 | 0.6412 | 0.3506 | 0.6879 | 2.2e-04 |
| office_4 | 107 | 80 | trudny | 10 | 0.4809 | 0.5388 | 0.6765 | 1.3365 | 1.4e-04 |

Wynik kontroli 1e-3: wszystkie zgodne.
