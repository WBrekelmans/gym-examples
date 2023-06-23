from gymnasium.envs.registration import register

register(
    id="gym_examples/EMS-v0",
    entry_point="gym_examples.envs:EnergyManagementEnv",
)

register(
    id="gym_examples/EMS_no_gen-v0",
    entry_point="gym_examples.envs:EnergyManagementEnv_no_gen",
)

register(
    id="gym_examples/EMS_no_gen-v2",
    entry_point="gym_examples.envs:EnergyManagementEnv_no_gen_V2",
)

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
)


