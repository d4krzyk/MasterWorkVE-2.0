"""Potok echo -> spektrogram: stale STFT i render_spectrogram().

Wydzielone z dawnego test_rlr_audio.py. Te same wartosci stoja za CALA
charakterystyka szumu, wiec zmiana ktorejkolwiek rozspoja historyczne pomiary
(GENERATOR_PARAMS.md §2, "Potok spektrogramu").
"""

import numpy as np
from scipy.signal import fftconvolve

SAMPLE_RATE = 44100
ECHO_MS = 60
ECHO_SAMPLES = int(round(SAMPLE_RATE * ECHO_MS / 1000))  # 2646 przy 44.1 kHz
STFT_N_FFT = 512
STFT_WIN_LENGTH = 64
STFT_HOP_LENGTH = 16
EXPECTED_SPEC_SHAPE = (2, 257, 166)
N_CHANNELS = 2  # binaural = 2 uszy; patrz uwaga w build_simulator()
BONUS_ANGLES = (0.0, 10.0, 90.0)
NAV_SEED = 42  # ustalony seed pathfindera dla powtarzalnosci wyniku


def render_spectrogram(rir, chirp):
    """RIR (n_samples, kanaly) x chirp (mono) -> magnituda STFT (kanaly, F, T).

    Wspoldzielone przez faze 5 (raport) i faze 7 (porownanie katow), zeby oba
    miejsca liczyly dokladnie to samo.
    """
    import librosa

    n_channels = rir.shape[1]
    echo = np.zeros((n_channels, ECHO_SAMPLES), dtype=np.float64)
    for ch in range(n_channels):
        convolved = fftconvolve(rir[:, ch], chirp, mode="full")
        n = min(ECHO_SAMPLES, convolved.shape[0])
        echo[ch, :n] = convolved[:n]
        # convolved krotszy niz 60ms zdarzyc sie moze tylko przy bardzo
        # krotkim RIR/chirpie - zerowy pad na koncu jest wtedy poprawnym,
        # cichym "brakiem echa" w tym oknie, nie artefaktem.

    spec = np.stack(
        [
            np.abs(
                librosa.stft(
                    echo[ch].astype(np.float32),
                    n_fft=STFT_N_FFT,
                    win_length=STFT_WIN_LENGTH,
                    hop_length=STFT_HOP_LENGTH,
                )
            )
            for ch in range(n_channels)
        ],
        axis=0,
    )
    # Uwaga: magnituda bez log1p - to wariant dla predykcji glebi (EchoNet),
    # nie wariant nawigacyjny z soundspaces/tasks/nav.py.
    return echo, spec

