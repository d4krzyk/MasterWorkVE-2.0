from pathlib import Path

import numpy as np
import pandas as pd

import habitat_sim

# 1. Konfiguracja symulatora. RLRAudio potrzebuje realnej geometrii sceny, żeby
# policzyć navmesh/collision mesh pod symulację propagacji dźwięku - pusta
# scena ("") kończy się błędem "non-existent mesh" niezwiązanym z tym, czy
# habitat-sim ma w ogóle wkompilowane wsparcie audio. Używamy więc tej samej
# sceny Replica co w test_replica.py / check_data.ipynb.
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "sound-spaces/data/scene_datasets/replica"
scene = "room_0"

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_dataset_config_file = str(data_path / "replica.scene_dataset_config.json")
cfg.scene_id = scene

# Audio to sensor, tak jak CameraSensorSpec - nie ma osobnego "audio managera"
# na Simulatorze. Musi być podpięty do agent_cfg.sensor_specifications JUŻ przy
# konstrukcji Simulatora: Simulator._sanitize_config() nadpisuje
# sim_cfg.create_renderer na podstawie obecności JAKIEGOKOLWIEK sensora u
# agenta (patrz simulator.py:_sanitize_config) - bez tego renderer zostaje
# wyłączony i ładowanie PTex (Replica) crashuje w
# PTexMeshData::getRenderingBuffer, zanim w ogóle dojdzie do testu audio.
audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
# Domyślny outputDirectory to "/home/AudioSimulation<N>" - katalog wprost pod
# /home, do którego zwykły użytkownik nie ma prawa zapisu. Ustawiamy własny,
# zapisywalny katalog.
audio_sensor_spec.outputDirectory = str(repo_root / "my-operations/audio_sim_output")
audio_sensor_spec.acousticsConfig.enableMaterials = False
audio_sensor_spec.channelLayout.channelType = (
    habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
)
audio_sensor_spec.channelLayout.channelCount = 2
audio_sensor_spec.acousticsConfig.sampleRate = 44100

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [audio_sensor_spec]

sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])

# sim = habitat_sim.Simulator(sim_cfg) rzuci wyjątkiem/abortem, jeśli
# habitat-sim nie zostało zbudowane z flagą --audio (patrz CLAUDE.md / build.sh)
has_audio = True
sim = None
try:
    sim = habitat_sim.Simulator(sim_cfg)
    print("SIM OK")

    # 2. Wczytanie realnych punktów nawigacyjnych z points.txt (grid kandydatów
    # ze skanu pomieszczenia, kolumny: id, a, b, c). Sprawdzone empirycznie na
    # graph.pkl tej samej sceny: habitat_x = a, habitat_z = -b (habitat_y,
    # czyli wysokość, nie jest tu zapisana - dostajemy ją z pathfindera przez
    # snap_point, co też potwierdza, że punkt faktycznie leży na navmeshu).
    points_path = repo_root / "my-operations/metadata/replica" / scene / "points.txt"
    points = pd.read_csv(points_path, sep="\t", header=None, names=["id", "a", "b", "c"])

    def point_to_xz(row):
        return float(row["a"]), -float(row["b"])

    bounds_min, _bounds_max = sim.pathfinder.get_bounds()
    y_guess = float(bounds_min[1]) + 0.1

    source_id, listener_id = 0, 100
    source_xz = point_to_xz(points.iloc[source_id])
    listener_xz = point_to_xz(points.iloc[listener_id])

    source_pos = sim.pathfinder.snap_point([source_xz[0], y_guess, source_xz[1]])
    listener_pos = sim.pathfinder.snap_point([listener_xz[0], y_guess, listener_xz[1]])

    assert sim.pathfinder.is_navigable(source_pos), f"punkt {source_id} nie leży na navmeshu"
    assert sim.pathfinder.is_navigable(listener_pos), f"punkt {listener_id} nie leży na navmeshu"

    print(f"Źródło dźwięku (points.txt id={source_id}):", source_pos)
    print(f"Słuchacz (points.txt id={listener_id}):", listener_pos)
    print("Odległość źródło-słuchacz [m]:", np.linalg.norm(np.array(source_pos) - np.array(listener_pos)))

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

    # ustaw agenta (a wraz z nim sensor audio, który jest jego dzieckiem w
    # scene graph) w punkcie słuchacza
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(listener_pos)
    sim.get_agent(0).set_state(agent_state)

    # źródło podnosimy na wysokość "uszu" (1.5 m), tak jak robi to
    # sound-spaces/soundspaces/continuous_simulator.py - listener dostaje tę
    # wysokość automatycznie z domyślnego offsetu AudioSensorSpec.position
    audio_sensor.setAudioSourceTransform(np.array(source_pos) + np.array([0.0, 1.5, 0.0]))
    audio_sensor.setAudioListenerTransform(
        audio_sensor.node.absolute_translation,
        np.array([1.0, 0.0, 0.0, 0.0]),  # tożsamościowy kwaternion (w, x, y, z)
    )

    audio_sensor.runSimulation(sim)
    ir = np.array(audio_sensor.getIR())

    print("AUDIO SENSOR ADDED OK")
    print("IR shape:", ir.shape)
    if ir.size > 0:
        print("IR max amplitude:", np.max(np.abs(ir)))
        print("IR non-zero samples:", int(np.count_nonzero(ir)))
    else:
        # Historycznie (submoduł rlr-audio-propagation przypięty na bdb262d,
        # lipiec 2022) audioSimulator_->UploadMesh() zawsze zwracał
        # ErrorCodes::MemoryAllocFailure (2018) - niezależnie od rozmiaru
        # siatki czy configu - i getIR() był zawsze pusty. Naprawione przez
        # podbicie submodułu do 4fd446b (patrz CLAUDE.md /
        # habitat-sim/local_changes.patch). Jeśli to się pojawi ponownie,
        # sprawdź `git -C habitat-sim/src/deps/rlr-audio-propagation log -1`.
        print(
            "UWAGA: pusta odpowiedź impulsowa - sprawdź wersję submodułu "
            "rlr-audio-propagation (patrz CLAUDE.md, sekcja o audio)."
        )
except Exception as e:
    has_audio = False
    print("AUDIO NOT SUPPORTED - build habitat-sim with --audio:", repr(e))

print("AUDIO SUPPORT FLAG:", has_audio)

if sim is not None:
    sim.close()
print("DONE")
