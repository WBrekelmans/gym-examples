import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
from ray.rllib.algorithms.algorithm import Policy
import tensorflow as tf
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym

# register and make the environment
select_env = "gym_examples/EMS_no_gen-v2"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen_V2())
env = gym.make(select_env)
# make algorithm
algo = PPOConfig().environment(select_env).build()

def get_agent_cost_iteration(foldername, iteration, env):
    chkpt_dst = "/home/willem/policies/" + foldername + "/checkpoint_" + str(iteration).zfill(6) + "/policies/default_policy"
    # load chkpt
    tf.compat.v1.enable_eager_execution()
    # https://github.com/tensorflow/tensorflow/issues/18304
    my_policy = Policy.from_checkpoint(chkpt_dst)
    # make arrays
    width_array = 8
    obs_array = []
    action_array = []
    energy_cost_array = []
    power_from_battery_array = []
    battery_percentage_vec = []
    # reset the environment
    obs, info = env.reset()
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array = np.empty((0, width_array))
    while (env.terminated == False):
        # determine action
        action = my_policy.compute_single_action([obs])[0]
        # step the environment
        output = env.step(action)
        # read out observatrions
        obs = output[0]
        obs = np.fromiter(obs.values(), dtype='float')
        obs_array = np.vstack((obs_array, obs))
        action_array = np.append(action_array, action)
        energy_cost_array = np.append(energy_cost_array, output[4]['energy_cost'])
        power_from_battery_array = np.append(power_from_battery_array, output[4]['power_from_battery'])

    obs_array_descaled['power_from_grid'] = env.descale_value(obs_array[:, 7], env.range_dict['power_from_grid'][0],
                                                              env.range_dict['power_from_grid'][1])

    obs_array_descaled['day_ahead_price'] = env.descale_value(obs_array[:, 4], env.range_dict['day_ahead_price'][0],
                                                              env.range_dict['day_ahead_price'][1])
    grid_cost = 0.25 * obs_array_descaled['power_from_grid'] * obs_array_descaled['day_ahead_price']
    total_grid_cost = sum(grid_cost)
    return total_grid_cost

agent_cost_arr = []
foldername='clip_and_scale_reward_proper_reset'
for i in range(1,75):
    agent_cost_arr = np.append(agent_cost_arr, get_agent_cost_iteration(foldername, i, env))
    print(i)
