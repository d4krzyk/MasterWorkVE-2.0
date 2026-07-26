import habitat_sim

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = "sound-spaces/data/scene_datasets/replica/replica.scene_dataset_config.json"
sim_cfg.scene_id = "room_0"

# habitat_sim.Simulator overrides create_renderer based on whether any agent
# has sensors (see simulator.py:_sanitize_config) - setting it directly here
# has no effect. Without a sensor, the renderer stays off and PTex (Replica)
# stage loading aborts in PTexMeshData::getRenderingBuffer.
sim_cfg.create_renderer = True

rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [128, 128]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor_spec]

cfg = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg]
)

sim = habitat_sim.Simulator(cfg)

print("OK - Replica loaded")
print("Navmesh loaded:", sim.pathfinder.is_loaded)

sim.close()
