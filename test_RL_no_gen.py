import os
import shutil
chkpt_root = "/home/willem/policies/v2_env"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# local mode
import ray
context = ray.init()
print('http://' + context.dashboard_url)

# import environment
from ray.tune.registry import register_env
from gym_examples.envs.EMG_no_gen_V2 import EnergyManagementEnv_no_gen_V2
select_env = "gym_examples/EMS_no_gen-v2"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen_V2())

# training configuration
from ray.rllib.algorithms.ppo import PPOConfig
config = (
    PPOConfig()
    .environment(select_env)
    .rollouts(num_rollout_workers=15)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [300, 300]})
    .evaluation(evaluation_num_workers=1)
    .training(train_batch_size=10000,  sgd_minibatch_size=100, gamma=0.9, lr=0.001, kl_coeff=0.3)
)

agent = config.build()

# train loop
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 200

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
