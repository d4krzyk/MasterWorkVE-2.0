import habitat_sim

print("IMPORT OK")

cfg = habitat_sim.SimulatorConfiguration()
cfg.load_semantic_mesh = False

cfg.scene_dataset_config_file = (
    "/home/d4krzyk/Dokumenty/MasterWorkVE/"
    "sound-spaces/data/scene_datasets/replica/"
    "replica.scene_dataset_config.json"
)

cfg.scene_id = (
    "/home/d4krzyk/Dokumenty/MasterWorkVE/"
    "sound-spaces/data/scene_datasets/replica/"
    "room_0/habitat/replica_stage.stage_config.json"
)

# habitat_sim.Simulator overrides create_renderer based on whether any agent
# has sensors (see simulator.py:_sanitize_config) - setting it directly here
# has no effect. Without a sensor, the renderer stays off and PTex (Replica)
# stage loading aborts in PTexMeshData::getRenderingBuffer. A sensor is
# required even for a load-only smoke test.
cfg.create_renderer = True
cfg.enable_physics = False

rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [128, 128]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor_spec]

print("CREATING SIM")

sim = habitat_sim.Simulator(
    habitat_sim.Configuration(cfg, [agent_cfg])
)

print("SIM OK")

sim.close()
