# Baseline dla sieci tylko-echo (opcjonalnie, CPU)

Zbiór treningowy B: 49464 próbek (36 orientacji), test@36: 6588 próbek, 96853180 ważnych pikseli; czas 47 s.

## Fakty (zmierzone)

- Średnia globalna głębi (ważne piksele treningu B): 1,7568 m.
- RMSE test@36 stałej mapy = średnia głębi **per piksel** z treningu B: 1,1026 m.
- RMSE test@36 **stałej liczby** (średnia globalna): 1,2186 m.
- RMSE test@36 (eval.json, średnia 3 ziaren): EB 0,5822 m, ESE 1,1629 m (seed 0: ESE 1,1647 m).
- ESE względem mapy per piksel: 0,0603 m (stosunek 1,055); względem stałej liczby: -0,0557 m (stosunek 0,954).
- EB względem mapy per piksel: -0,5204 m (stosunek 0,528).
- Średnie (po pikselach) odchylenie standardowe wartości między próbkami testowymi, per piksel (predykcje seed 0, float32 na CPU, prawdziwe echo): EB 0,9452 m, ESE 0,0108 m; prawda: 1,0691 m.

## Hipotezy (niezweryfikowane tym pomiarem)

- Jeśli RMSE ESE jest bliskie lub większe niż stałej mapy, ESE nie wyciąga z echa informacji o scenie i jej wynik odpowiada predykcji niezależnej od próbki (bramka zadziałała).
- Małe zróżnicowanie predykcji ESE między próbkami (std) względem EB świadczyłoby, że ESE zwija się do mapy średniej; sama liczba tego nie dowodzi (mapa średnia vs. predykcja zależna od próbki, ale błędna, dają różny RMSE przy podobnym std).
- Uwaga: predykcje liczone w float32 na CPU, a RMSE w eval.json z AMP na GPU; różnica numeryczna nie była osobno mierzona.
