"""Jeden dlugo zyjacy Simulator na scene + pojedynczy render echolokacji."""

import time

import numpy as np

from . import audio, spectrogram
from .params import (INDIRECT_RAY_COUNT, SENSOR_HEIGHT, THREAD_COUNT, WARMUP_DISCARD)
from .paths import CHIRP_PATH, MATERIAL_CONFIG, OUT_ROOT, scene_mesh
from .store import SPEC_SHAPE

class Renderer:
    """Jeden dlugo zyjacy Simulator + jeden render = jedna probka RIR.

    Sciezka renderowania jest CELOWO wywolaniem `echo_core.audio.phase3_echolocation`,
    a nie wlasna kopia: to dokladnie ta funkcja, ktora wygenerowala cala
    charakterystyke szumu (diagnostics/common.py wola ja przez render_raw()).
    Przepisanie jej tutaj groziloby cicha zmiana kolejnosci wywolan audio,
    a wiec i sekwencji RNG, wzgledem ktorej skalibrowano SIGNAL_10DEG i rozklad N.
    """

    def __init__(self, scene, log):
        # audio + spectrogram zamiast dawnego "import test_rlr_audio as tra"


        self.log = log
        self.scene = scene

        class _Args:
            pass

        args = _Args()
        args.scene = str(scene_mesh(scene))
        args.sensor_height = SENSOR_HEIGHT
        args.material_config = str(MATERIAL_CONFIG)
        args.out_dir = str(OUT_ROOT / "_rlr_scratch")
        args.indirect_ray_count = INDIRECT_RAY_COUNT
        args.thread_count = THREAD_COUNT
        args.gpu_device_id = 0

        t0 = time.perf_counter()
        self.sim = audio.build_simulator(args)
        log.info("Simulator zbudowany w %.1f s (%s, %d promieni, %d watek)",
                 time.perf_counter() - t0, scene, INDIRECT_RAY_COUNT, THREAD_COUNT)

        import librosa
        self.chirp, _sr = librosa.load(str(CHIRP_PATH), sr=spectrogram.SAMPLE_RATE, mono=True)

        # setAudioMaterialsJSON() musi paść PRZED pierwszym runSimulation (to ono
        # wola loadSemanticMesh, ktore zamyka baze materialow). Kazde kolejne
        # wywolanie jest w AudioSensor.cpp:173-182 no-opem konczacym sie
        # ostrzezeniem w logu, wiec podajemy config tylko przy pierwszym
        # renderze — zachowanie symulatora identyczne, log krotszy o ~600 tys. linii.
        self._materials_pending = True
        self.n_renders = 0
        self.n_warmup = 0
        self.render_seconds = 0.0

    def warmup(self, position, n=WARMUP_DISCARD):
        """Rendery rozgrzewkowe — wykonane i ODRZUCONE, patrz WARMUP_DISCARD.

        Wykonujemy je na pozycji pierwszej lokalizacji, bo i tak trzeba gdzies
        stanac, a rozgrzewka jest wlasnoscia instancji Simulatora, nie pozycji
        (zmierzone: po przeniesieniu agenta efekt sie NIE powtarza).
        """
        if n <= 0:
            return
        t0 = time.perf_counter()
        for _ in range(n):
            self.render(position, 0.0)
        self.n_warmup = n
        self.log.info("Rozgrzewka: %d renderow odrzuconych w %.1f s "
                      "(pierwsze ~10 renderow instancji ma szum wyzszy o 10-20 %%)",
                      n, time.perf_counter() - t0)

    def render(self, position, angle_deg):
        """-> (spec float32 (2,257,166), rgb uint8, depth float32)"""
        mc = str(MATERIAL_CONFIG) if self._materials_pending else None
        t0 = time.perf_counter()
        # run_simulation=False: symulacje akustyczna uruchamia raz
        # get_sensor_observations() (przez Sensor._get_audio_observation()) i to
        # jej wynik trafia do obserwacji. Jawne runSimulation() liczylo druga,
        # nieodczytywana — polowa czasu renderu szla do kosza. Rownowaznosc obu
        # sciezek zweryfikowana pomiarowo 2026-07-28 na dwoch skrajnych pozycjach
        # (najglosniejsza i najcichsza zmierzona): wszystkie roznice ponizej
        # 1.3 SE, N z reguly 30->31 i 4->4. Szczegoly: GENERATOR_PARAMS.md §4.3.
        obs, _listener, _rot = audio.phase3_echolocation(
            self.sim, position, float(angle_deg), mc, run_simulation=False)
        self._materials_pending = False

        rir = np.transpose(np.array(obs["audio_sensor"]))
        if rir.size == 0 or not np.any(rir):
            raise RuntimeError(
                f"RIR to same zera dla pozycji {position} / kata {angle_deg} — symulacja akustyczna "
                "nie zwrocila echa (patrz smoke_test_rlr_audio.phase4_validate_rir)")
        _echo, spec = spectrogram.render_spectrogram(rir, self.chirp)
        if spec.shape != SPEC_SHAPE:
            raise RuntimeError(f"spektrogram ma ksztalt {spec.shape}, oczekiwano {SPEC_SHAPE}")

        # np.flip() w Sensor.get_observation() zwraca WIDOK na bufor sensora,
        # ktory kolejny render nadpisze — stad jawna kopia.
        rgb = np.array(obs["rgb"], dtype=np.uint8, copy=True)
        depth = np.array(obs["depth"], dtype=np.float32, copy=True)

        self.n_renders += 1
        self.render_seconds += time.perf_counter() - t0
        return spec.astype(np.float32, copy=False), rgb, depth

    def close(self):
        try:
            self.sim.close()
        finally:
            self.sim = None


