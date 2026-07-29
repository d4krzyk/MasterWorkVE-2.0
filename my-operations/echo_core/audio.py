"""Symulator habitat-sim z sensorem audio RLR + jeden render echolokacji.

Wydzielone z dawnego test_rlr_audio.py BEZ zmiany tresci: to dokladnie te
funkcje, ktore wygenerowaly cala charakterystyke szumu, wiec kazda zmiana
kolejnosci wywolan audio zmienialaby sekwencje RNG.
"""

from pathlib import Path

import numpy as np

# Ta kolejnosc importow (quaternion przed habitat_sim) jest wymagana przez
# lokalny patch tego repo - patrz habitat-sim/local_changes.patch / CLAUDE.md,
# inaczej apka konczy sie "free(): invalid pointer" (upstream habitat-sim#1747).
import quaternion  # noqa: F401
import habitat_sim

from .spectrogram import N_CHANNELS, SAMPLE_RATE


class PhaseFailure(Exception):
    """Blad konkretnej fazy - odrozniony od bledow programistycznych."""


def build_simulator(args):
    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise PhaseFailure(f"Plik sceny nie istnieje: {scene_path}")

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_path)
    # Scena bez semantic mesh (baseline) - AudioSensor::loadMesh() (branch
    # nieuzywajacy materialow) i tak uzywa tylko geometrii render-mesha, nie
    # semantycznej, wiec nie ma potrzeby ladowac osobnych semantic assetow
    # tutaj. Jedyny wyjatek to --material-config, patrz nizej.
    cfg.load_semantic_mesh = False
    cfg.enable_physics = False
    # Bez sensora habitat_sim.Simulator._sanitize_config() wylacza renderer i
    # ladowanie PTex/mesh krysuje w PTexMeshData::getRenderingBuffer zanim w
    # ogole dojdzie do audio - create_renderer=True jest wiec wymagane nawet
    # jesli interesuje nas tylko dzwiek.
    cfg.create_renderer = True
    # Jawnie, mimo ze 0 jest wartoscia domyslna: generate_echo_dataset.py zapisuje
    # ten parametr do atrybutow HDF5 jako czesc opisu reprodukowalnosci, wiec nie
    # moze zalezec od domyslnej wartosci biblioteki (por. blad 1.25 vs 1.5 m
    # opisany w PKL_FORMAT.md, gdzie poleganie na domyslnej wysokosci kamery
    # rozjechalo odtworzenie datasetu).
    cfg.gpu_device_id = int(getattr(args, "gpu_device_id", None) or 0)

    use_materials = args.material_config is not None
    if use_materials:
        # setAudioMaterialsJSON() ma efekt TYLKO na sciezce loadSemanticMesh()
        # (patrz habitat-sim/src/esp/sensor/AudioSensor.cpp:136-146), ktora
        # wymaga jednoczesnie acousticsConfig.enableMaterials=True ORAZ
        # zaladowanej semantycznej sceny (sim.semanticSceneExists()). Bez
        # load_semantic_mesh=True material-config zostalby po cichu
        # zignorowany.
        cfg.load_semantic_mesh = True

    # Wysokosc wszystkich sensorow nad wezlem agenta. Domyslne 1.5 m zachowane dla
    # zgodnosci wstecznej ze WSZYSTKIMI wczesniejszymi pomiarami szumu (E1-E4,
    # checkpoint-boundary, Blok A/B/C) - zmiana domyslnej rozspoinilaby je.
    # Produkcja idzie na 1.25 m: tyle ma kamera odtworzona z scene_observations_128.pkl
    # (patrz PKL_FORMAT.md), a agent ucielesniony musi widziec i slyszec z jednego
    # punktu - roznica 25 cm zmienia echo mocniej niz obrot o 10 stopni (patrz
    # eksperyment listener_height).
    sensor_height = float(getattr(args, "sensor_height", None) or 1.5)

    # hfov jawnie, z tego samego powodu co gpu_device_id wyzej: 90 stopni to
    # wartosc odtworzona wstecznie z scene_observations_128.pkl (PKL_FORMAT.md,
    # kontrola negatywna przy 70 stopniach: RGB RMSE 33.59 zamiast 0.0077), a nie
    # dowolna. Rowna sie akurat domyslnej CameraSensorSpec, wiec nic to nie zmienia
    # dla wczesniejszych pomiarow.
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [128, 128]
    rgb_spec.position = [0.0, sensor_height, 0.0]
    rgb_spec.hfov = 90.0

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [128, 128]
    depth_spec.position = [0.0, sensor_height, 0.0]
    depth_spec.hfov = 90.0

    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.outputDirectory = str(Path(args.out_dir) / "rlr_sim_output")
    audio_spec.position = [0.0, sensor_height, 0.0]
    audio_spec.acousticsConfig.sampleRate = SAMPLE_RATE
    audio_spec.acousticsConfig.enableMaterials = use_materials
    # Domyslnie 500 promieni na jednym watku - taka konfiguracja stoi za CALA
    # dotychczasowa charakteryzacja szumu (E1/E3/E4, checkpoint-boundary), wiec
    # zmiana domyslnej wartosci rozspoinilaby te pomiary. Oba parametry mozna
    # jednak nadpisac przez args, bo E2 porownuje wlasnie "wiecej promieni" z
    # "wiecej renderow", a to wymaga przemiatania indirectRayCount.
    # UWAGA: swiezy AudioSensorSpec() raportuje indirectRayCount=5000 - to
    # domyslna wartosc biblioteki, NIE ta uzywana tutaj.
    audio_spec.acousticsConfig.indirectRayCount = getattr(args, "indirect_ray_count", None) or 500
    audio_spec.acousticsConfig.threadCount = getattr(args, "thread_count", None) or 1
    audio_spec.channelLayout.channelType = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    # Binaural = 2 kanaly (lewe/prawe ucho) z definicji - channelCount=1
    # bylby niespojny z tym layoutem i dalby zly ksztalt na wejsciu do
    # spektrogramu (faza 5 zaklada dokladnie 2 kanaly).
    audio_spec.channelLayout.channelCount = N_CHANNELS

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec, audio_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    return sim



def phase3_echolocation(sim, position, angle_deg, material_config, run_simulation=True):
    """Echolokacja: ustawia poze, zrodlo i sluchacza, zwraca obserwacje.

    `run_simulation` steruje JEDNA linia (jawnym `audio_sensor.runSimulation()`)
    i istnieje, bo ta linia jest zbedna:

    `sim.get_sensor_observations()` dla sensora typu AUDIO wchodzi w
    `Sensor._get_audio_observation()` (habitat-sim/src_python/habitat_sim/
    simulator.py:763-777), ktore samo ustawia transform sluchacza, wola
    `runSimulation()` i dopiero jego wynik zwraca przez `getIR()`. Jawne
    wywolanie wyzej liczy wiec CALA symulacje akustyczna, ktorej wynik nikt nie
    odczytuje — 50 % czasu renderu idzie do kosza (zmierzone: 283.8 ms wobec
    143.2 ms na `office_1`, dokladnie 2x).

    Domyslne `True` zachowuje zachowanie historyczne, na ktorym oparta jest CALA
    charakterystyka szumu (`diagnose_rlr_noise.py` wola te funkcje przez
    `render_raw()`), zeby tamte pomiary pozostaly odtwarzalne co do bitu.
    Generator datasetu podaje jawnie `False` — rownowaznosc obu sciezek
    zweryfikowano pomiarowo 2026-07-28, patrz GENERATOR_PARAMS.md §4.3.

    NIE usuwamy `setAudioSourceTransform()` (nizej): `_get_audio_observation()`
    ustawia wylacznie transform SLUCHACZA, wiec bez tamtej linii zrodlo dzwieku
    nigdy nie zostaloby ustawione i echolokacja (zrodlo wspollokowane
    z odbiornikiem) przestalaby dzialac.
    """
    agent_state = habitat_sim.AgentState()
    agent_state.position = position
    agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        np.deg2rad(angle_deg), np.array([0.0, 1.0, 0.0])
    )
    sim.get_agent(0).set_state(agent_state)

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

    if material_config is not None:
        audio_sensor.setAudioMaterialsJSON(material_config)

    # Echolokacja: zrodlo dzwieku WSPOLLOKOWANE z odbiornikiem - agent emituje
    # chirp z wlasnej pozycji (na wysokosci uszu, stad node.absolute_translation
    # zamiast surowego position agenta) i nasluchuje wlasnego echa.
    listener_pos = np.array(audio_sensor.node.absolute_translation)
    quat_arr = quaternion.as_float_array(agent_state.rotation)  # [w, x, y, z]

    audio_sensor.setAudioSourceTransform(listener_pos)
    audio_sensor.setAudioListenerTransform(listener_pos, quat_arr)
    if run_simulation:
        # Zbedne — get_sensor_observations() nizej i tak uruchomi symulacje
        # i to JEJ wynik zwroci. Patrz docstring.
        audio_sensor.runSimulation(sim)

    obs = sim.get_sensor_observations()
    return obs, listener_pos, agent_state.rotation

