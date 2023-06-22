# libraries
import os
import shutil
import ray
from ray import air, tune
from ray.tune.registry import register_env
from gym_examples.envs.EMS_no_gen import EnergyManagementEnv_no_gen

# to store results
chkpt_root = "/home/willem/policies/test"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# environment
select_env = "gym_examples/EMS_no_gen-v0"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen())

# initialize ray
ray.init(include_dashboard=False)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .environment(select_env)
    .build()
)

# grid search
config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(),
    param_space=config,
)

tuner.fit()
ray.shutdown()
