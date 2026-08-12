"""Metryki zadania pretekstowego -- i pulapka metryczna, ktora trzeba obejsc.

TRAFNOSC TOP-1 NIE JEST POROWNYWALNA MIEDZY ROZNYMI K. Poziom losowy spada z
25 % przy K=4 do 2,8 % przy K=36, wiec sam spadek trafnosci nic nie mowi o tym,
czy zadanie zostalo rozwiazane gorzej. Metryka porownywalna to SREDNI
BEZWZGLEDNY BLAD KATOWY (MAAE), liczony jako odleglosc PO OKREGU miedzy klasa
przewidziana a prawdziwa. Dla rozkladu rownomiernego po okregu MAAE poziomu
losowego wynosi dokladnie 90 stopni NIEZALEZNIE OD K -- wszystkie warianty maja
wiec wspolny, staly punkt odniesienia.

(Dowod na 90 stopni: przy roznicy katow rozlozonej rownomiernie po okregu
odleglosc po okregu jest rozlozona rownomiernie na [0, 180], wiec ma srednia 90.
Dla siatki dyskretnej K rownomiernych klas srednia zbioru odleglosci
{0, s, 2s, ..., 180, ..., s} rowniez wynosi 90.)

Raportowane sa wszystkie cztery wielkosci z punktu 4.3:
  * MAAE                       -- podstawowa, porownywalna miedzy K
  * trafnosc w tolerancji      -- +/-10, +/-30, +/-45 stopni
  * trafnosc top-1 doslowna    -- WYLACZNIE do zestawienia z 66 % Gao przy K=4
  * macierz pomylek            -- oczekiwanie: wiekszosc bledow miedzy klasami
                                  sasiednimi

Oraz rozbicie z punktu 4.6: MAAE osobno dla par o prawdziwym przesunieciu
<= 20 stopni i > 20 stopni. Przy K=36 rozroznienie przesuniecia 0 od 10 stopni
wymaga sygnalu o RMSE ok. 0,0644 przy szumie probki po usrednieniu ok. 0,018
(SNR ok. 3,5) -- jest nad szumem, ale niewiele. Jesli siec jest bezradna ponizej
20 stopni, to jest wynik sam w sobie i wyznacza faktyczna granice
rozdzielczosci metody.
"""

from __future__ import annotations

import numpy as np

# Tolerancje raportowane obok MAAE (w stopniach).
TOLERANCES_DEG = (10, 30, 45)
# Granica rozbicia z punktu 4.6.
FINE_SHIFT_LIMIT_DEG = 20
CHANCE_MAAE_DEG = 90.0


def circular_error_deg(pred_cls: np.ndarray, true_cls: np.ndarray, k: int) -> np.ndarray:
    """Odleglosc po okregu miedzy klasami, w stopniach, w zakresie [0, 180]."""
    step = 360.0 / k
    d = np.abs(pred_cls.astype(np.int64) - true_cls.astype(np.int64)) % k
    d = np.minimum(d, k - d)
    return d * step


class PretextEvaluator:
    """Akumuluje predykcje i liczy caly zestaw metryk 4.3 + 4.6."""

    def __init__(self, k: int):
        self.k = k
        self.step_deg = 360.0 / k
        self._pred: list[np.ndarray] = []
        self._true: list[np.ndarray] = []

    def update(self, logits, labels) -> None:
        p = logits.detach().argmax(dim=1).cpu().numpy()
        t = labels.detach().cpu().numpy()
        self._pred.append(p)
        self._true.append(t)

    def result(self) -> dict:
        if not self._pred:
            return {"n": 0}
        pred = np.concatenate(self._pred)
        true = np.concatenate(self._true)
        err = circular_error_deg(pred, true, self.k)
        true_shift = true.astype(np.float64) * self.step_deg
        # Przesuniecie prawdziwe jako odleglosc po okregu od zera: 350 stopni to
        # obrot o 10 stopni w druga strone, wiec "drobne" przesuniecie.
        true_shift_circ = np.minimum(true_shift, 360.0 - true_shift)

        out: dict = {
            "n": int(pred.size),
            "K": self.k,
            "step_deg": self.step_deg,
            "MAAE_deg": float(err.mean()),
            "MAAE_chance_deg": CHANCE_MAAE_DEG,
            "MAAE_vs_chance_pct": float(100.0 * (1 - err.mean() / CHANCE_MAAE_DEG)),
            "median_AE_deg": float(np.median(err)),
            "top1": float((pred == true).mean()),
            "top1_chance": 1.0 / self.k,
        }
        for tol in TOLERANCES_DEG:
            out[f"acc_within_{tol}deg"] = float((err <= tol).mean())
            # Poziom losowy tolerancji zalezy od K -- podajemy go obok, zeby
            # liczby nie wygladaly lepiej, niz sa, przy duzym K.
            n_in = int(np.sum(circular_error_deg(np.arange(self.k), np.zeros(self.k, int),
                                                 self.k) <= tol))
            out[f"acc_within_{tol}deg_chance"] = n_in / self.k

        # 4.6 -- czy zadanie jest w ogole wykonalne przy najdrobniejszej granulacji.
        fine = true_shift_circ <= FINE_SHIFT_LIMIT_DEG
        coarse = ~fine
        out["by_true_shift"] = {
            f"fine_le_{FINE_SHIFT_LIMIT_DEG}deg": {
                "n": int(fine.sum()),
                "MAAE_deg": float(err[fine].mean()) if fine.any() else float("nan"),
                "top1": float((pred[fine] == true[fine]).mean()) if fine.any() else float("nan"),
            },
            f"coarse_gt_{FINE_SHIFT_LIMIT_DEG}deg": {
                "n": int(coarse.sum()),
                "MAAE_deg": float(err[coarse].mean()) if coarse.any() else float("nan"),
                "top1": float((pred[coarse] == true[coarse]).mean()) if coarse.any() else float("nan"),
            },
        }

        cm = np.zeros((self.k, self.k), dtype=np.int64)
        np.add.at(cm, (true, pred), 1)
        out["confusion_matrix"] = cm.tolist()
        # Skrot macierzy pomylek: jaki udzial bledow trafia w klase SASIEDNIA.
        wrong = pred != true
        if wrong.any():
            e = circular_error_deg(pred[wrong], true[wrong], self.k)
            out["errors_to_adjacent_class_fraction"] = float((e <= self.step_deg + 1e-9).mean())
        else:
            out["errors_to_adjacent_class_fraction"] = float("nan")
        return out
