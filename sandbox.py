from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# environment
select_env = "gym_examples/EMS_no_gen-v0"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen())

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=select_env)
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")