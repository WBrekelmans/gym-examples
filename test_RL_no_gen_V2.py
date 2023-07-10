run_name = "test"

# make folders
import os
import shutil
chkpt_root = "/home/willem/policies/" + run_name
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results_str = "{}/ray_results/" + run_name + "/"
ray_results_root = ray_results_str.format(os.getenv("HOME"))
shutil.rmtree(ray_results_root, ignore_errors=True, onerror=None)

# turn on ray and make dashboard
import ray

context = ray.init()
print('http://' + context.dashboard_url)

# import environment
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2

# training configuration
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(env=EnergyManagementEnv_no_gen_V2, clip_actions=True, env_config={'env_config': {}})
    .rollouts(num_rollout_workers=14)
    .training(model={"fcnet_hiddens": [64, 64]}, gamma=0.99, lr=0.00001)
    .debugging(log_level='INFO')
)

# make agent
agent = config.build()

# train loop
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 1

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
