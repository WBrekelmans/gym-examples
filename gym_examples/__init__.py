from gymnasium.envs.registration import register

register(
    id="gym_examples/EMS-v0",
    entry_point="gym_examples.envs:EnergyManagementEnv",
    max_episode_steps=300,
)

from gymnasium.envs.registration import register

register(
    id="gym_examples/EMS_no_gen-v0",
    entry_point="gym_examples.envs:EnergyManagementEnv_no_gen",
    max_episode_steps=300,
)

from gymnasium.envs.registration import register

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)


