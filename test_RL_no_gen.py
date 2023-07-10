import os
import shutil
run_name = "special_reward_chkpt_test"
chkpt_root = "/home/willem/policies/" + run_name
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results_str = "{}/ray_results/" + run_name + "/"
ray_results = ray_results_str.format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# local mode
import ray
context = ray.init()
print('http://' + context.dashboard_url)

# import environment
from ray.tune.registry import register_env
import gymnasium as gym
env = gym.make("gym_examples/EMS_no_gen-v2")
register_env("selected_env", lambda config: env)


# training configuration
from ray.rllib.algorithms.ppo import PPOConfig
config = (
    PPOConfig()
    .environment(env="selected_env", clip_actions = True)
    .rollouts(num_rollout_workers=14)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [32, 32]}, gamma=0.995, lr=0.0001, lambda_=0.96)
    .debugging(log_level='INFO')
)
#     .evaluation(evaluation_num_workers=1, evaluation_interval=100, evaluation_duration=1, evaluation_duration_unit='episodes')
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
