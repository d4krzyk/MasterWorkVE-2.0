"""echo_core — wspolny rdzen generatora ech, diagnostyki i pulpitu.

Podzial modulow:
    paths        sciezki projektu (jedyne zrodlo prawdy o ukladzie katalogow)
    params       parametry produkcyjne z GENERATOR_PARAMS.md
    scenes       kolejnosc scen + zrodlo pozycji lokalizacji
    spectrogram  stale STFT + echo -> spektrogram
    audio        Simulator z sensorem RLR + jeden render echolokacji
    noise        estymatory sigma_1 / SNR i regula wyznaczajaca N
    renderer     dlugo zyjacy Simulator na scene, z rozgrzewka
    store        HDF5: uklad pol, atrybuty reprodukowalnosci, zapis i wznawianie
    verify       --verify (bez GPU)
    status       --status (bez GPU)
    runtime      sygnaly, logowanie, formatery
"""
