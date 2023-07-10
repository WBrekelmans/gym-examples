import ray
from ray import air, tune

ray.init()

from ray.rllib.algorithms.ppo import PPOConfig
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2

config = (
    PPOConfig()
    .environment(env=EnergyManagementEnv_no_gen_V2, clip_actions=True, env_config={'env_config': {}})
    .rollouts(num_rollout_workers=13)
    .training(model={"fcnet_hiddens": tune.grid_search([[32, 32], [64, 64], [128, 128]])}, gamma=tune.grid_search([0.9, 0.925, 0.95]),
              lr=tune.grid_search([0.01, 0.00055, 0.0001]))
    .debugging(log_level='INFO')
)

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"training_iteration": 500},
    ),
    param_space=config,
)

tuner.fit()
