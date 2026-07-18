import habitat_sim
from habitat_sim.sensor import RLRAudioPropagationConfiguration

# 1. Konfiguracja symulatora (pusty świat)
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = ""  # brak sceny = empty stage

agent_cfg = habitat_sim.AgentConfiguration()

sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])

sim = habitat_sim.Simulator(sim_cfg)

print("SIM OK")

# 2. RLRAudio config
audio_cfg = RLRAudioPropagationConfiguration()

print("AUDIO CONFIG CREATED")
print(audio_cfg)

# 3. Sprawdzenie czy audio system jest dostępny
has_audio = hasattr(sim, "get_audio_manager") or hasattr(sim, "audio_manager")

print("AUDIO SUPPORT FLAG:", has_audio)

sim.close()
print("DONE")