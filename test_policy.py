import numpy as np

from ray.rllib.policy.policy import Policy

# Use the `from_checkpoint` utility of the Policy class:
chkpt_dst = "/home/willem/policies/exb/checkpoint_000050/policies/default_policy"
my_policy = Policy.from_checkpoint(chkpt_dst)

# grab environment
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
env = gym.make('gym_examples/EMS-v0')
env = FlattenObservation(env)

# reset the environment
obs, info = env.reset()
action = my_policy.compute_single_action(obs)[0]
print(action)

# first action


# further actions
action = my_policy.compute_single_action(obs)
env.step(action)

obs, info = my_policy.compute_single_action(env.step(action))

i = 1
while env.terminated != True:
    print(env.step(env.action_space.sample()))
    print(i)
    i += 1

obs = np.array([0.0, 0.1, 0.2, 0.3])  # individual CartPole observation
action = my_restored_policy.compute_single_action(obs)

print(f"Computed action {action} from given CartPole observation.")