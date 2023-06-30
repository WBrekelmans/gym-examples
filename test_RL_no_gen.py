import os
import shutil
chkpt_root = "/home/willem/policies/clip_and_scale_reward_proper_reset"
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
from ray.rllib.algorithms.ppo import PPOConfig
config = (
    PPOConfig()
    .environment(env="selected_env", clip_actions = True)
    .rollouts(num_rollout_workers=3)
    .framework("tf2")
    .evaluation(evaluation_num_workers=1, evaluation_interval=100, evaluation_duration=5, evaluation_duration_unit='episodes')
    .training(model={"fcnet_hiddens": [32, 32]}, train_batch_size=10000,  sgd_minibatch_size=100, gamma=0.9, lr=0.001, kl_coeff=0.3)
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
