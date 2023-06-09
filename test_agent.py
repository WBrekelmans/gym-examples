# https://github.com/ray-project/ray/issues/7983

# inputs
chkpt_dst = "/home/willem/PycharmProjects/gym-examples/tmp/exa/checkpoint_000050"

# load environment
from ray.tune.registry import register_env
from gym_examples.envs.EMS import EnergyManagementEnv
select_env = "gym_examples/EMS-v0"
register_env(select_env, lambda config: EnergyManagementEnv())

# config agent
from ray.rllib.algorithms.ppo import PPOConfig
config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(select_env)
    .rollouts(num_rollout_workers=2)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

# make agent
# https://github.com/tensorflow/tensorflow/issues/18304
import ray
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
ray.init(ignore_reinit_error=True)
agent = config.build()

# load agent
ppo = Algorithm.from_checkpoint(chkpt_dst)
