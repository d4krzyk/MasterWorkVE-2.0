"""Estymatory szumu i regula wyznaczajaca N.

Ten sam podzial na dwie rozlaczne polowki, ktorego uzywa kazdy eksperyment
w projekcie — patrz GENERATOR_PARAMS.md §3.2.
"""

import numpy as np

from .params import N_MAX, N_MIN, SIGNAL_10DEG, TARGET_SNR

# ktorego uzywa kazdy eksperyment w projekcie (diagnose_rlr_noise.py:2584-2591,
# 2688-2690) i ktory definiuje GENERATOR_PARAMS.md §3.2.
# ---------------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mean_f32(specs):
    """Srednia po renderach w float32.

    GENERATOR_PARAMS.md §4.1, regula bezwzgledna: usredniac w float32, rzutowac
    na float16 dopiero na gotowym wyniku. Akumulacja w float16 kumulowalaby blad
    zaokraglenia przy kazdym z N dodawan i zweryfikowane liczby o bledzie
    kwantyzacji (7-8e-5 RMSE, 112-240x ponizej podlogi szumu) przestalyby
    obowiazywac.
    """
    return np.mean(np.stack(specs, axis=0), axis=0, dtype=np.float32)


def sigma_1_from_specs(specs):
    """Szum POJEDYNCZEGO renderu, oszacowany z podzialu `specs` na dwie polowki.

        sigma_1 = RMSE(A, B) / sqrt(2) * sqrt(h)

    gdzie A, B to srednie z h = len(specs)//2 rozlacznych renderow kazda.
    Dzielenie przez sqrt(2) bierze sie stad, ze RMSE dwoch niezaleznych estymat
    o szumie sigma_h wynosi sqrt(2)*sigma_h; mnozenie przez sqrt(h) przelicza
    szum estymaty z h renderow na szum pojedynczego renderu.

    Przy nieparzystym len(specs) ostatni render nie wchodzi do PODZIALU (bo
    polowki musza byc rownoliczne), ale wchodzi do estymaty koncowej — patrz
    snr_from_specs().
    """
    n = len(specs)
    h = n // 2
    if h < 1:
        raise ValueError(f"potrzeba >= 2 renderow do podzialu na polowki, jest {n}")
    a = _mean_f32(specs[:h])
    b = _mean_f32(specs[h:2 * h])
    return _rmse(a, b) / np.sqrt(2.0) * np.sqrt(h), h


def snr_from_specs(specs, signal=SIGNAL_10DEG):
    """SNR estymaty zbudowanej ze WSZYSTKICH `len(specs)` renderow.

    Estymata koncowa usrednia n renderow, wiec jej szum to sigma_n =
    sigma_1/sqrt(n), i to on stoi w mianowniku:

        snr = SIGNAL_10DEG / sigma_n = SIGNAL_10DEG * sqrt(n) / sigma_1

    UWAGA — odstepstwo od doslownego zapisu w GENERATOR_PARAMS.md §3.4.1.
    Dokument podaje tam `snr = SIGNAL_10DEG / (RMSE(A,B)/sqrt(2))`. To wyrazenie
    jest o czynnik sqrt(2) za male: RMSE(A,B)/sqrt(2) to szum estymaty z POLOWY
    renderow (sigma_{n/2}), a nie z wszystkich n. Przy dokladnie trafionym N
    dawaloby snr = 3.5/sqrt(2) = 2.47, czyli ponizej progu ZAWSZE — petla
    weryfikacyjna z §3.4 dorenderowywalaby kazda probke, co przeczy zdaniu tego
    samego paragrafu, ze "kosztuje tylko dorenderowanie nielicznych przypadkow".
    Wersja uzyta tutaj jest samouzgodniona z regula na N z §3.2: jesli
    n = (3.5*sigma_1/SIGNAL)^2, to sigma_n = SIGNAL/3.5, czyli snr = 3.5 dokladnie.
    Jest to tez ta sama dekompozycja, ktora zapisano w komentarzach
    diagnose_rlr_noise.py:1764-1765 i 2026-2027 ("estymata full ma 2N renderow,
    wiec jej szum to sigma_N/sqrt(2) = noise/2").
    """
    n = len(specs)
    sigma_1, _h = sigma_1_from_specs(specs)
    sigma_n = sigma_1 / np.sqrt(n)
    snr = float(signal / sigma_n) if sigma_n > 0 else float("inf")
    return snr, float(sigma_1)


def plan_n(sigma_1):
    """sigma_1 -> (n_raw, n_planned, clamped) wg GENERATOR_PARAMS.md §3.2."""
    n_raw = int(np.ceil((TARGET_SNR * sigma_1 / SIGNAL_10DEG) ** 2)) if sigma_1 > 0 else N_MIN
    n_planned = int(min(max(n_raw, N_MIN), N_MAX))
    clamped = "min" if n_raw < N_MIN else ("max" if n_raw > N_MAX else "")
    return n_raw, n_planned, clamped

