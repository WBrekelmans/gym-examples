import os
import shutil
chkpt_root = "/home/willem/policies/week_with_gen_20"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# local mode
import ray
ray.init(ignore_reinit_error=True)

from ray.tune.registry import register_env
from gym_examples.envs.EMS import EnergyManagementEnv
select_env = "gym_examples/EMS_no_gen-v0"
register_env(select_env, lambda config: EnergyManagementEnv())

from ray.rllib.algorithms.ppo import PPOConfig
config = (
    PPOConfig()
    .environment(select_env)
    .rollouts(num_rollout_workers=2)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

agent = config.build()

# train loop
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 20

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
