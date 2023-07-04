import os
import shutil
chkpt_root = "/home/willem/policies/TD3"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# local mode
import ray
context = ray.init()
print('http://' + context.dashboard_url)

# import environment
from ray.tune.registry import register_env
import gymnasium as gym
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
env = gym.make("gym_examples/EMS_no_gen-v2")
register_env("selected_env", lambda config: env)

# training configuration
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.td3 import TD3Config

config = (
    TD3Config()
    .environment(env="selected_env", clip_actions = True)
    .rollouts(num_rollout_workers=8)
    .debugging(log_level='INFO')
)

agent = config.build()

# train loop
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 500

for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))
