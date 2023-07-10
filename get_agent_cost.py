import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
from ray.rllib.algorithms.algorithm import Policy
from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym

# load the environment
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
env = EnergyManagementEnv_no_gen_V2(env_config={'env_config': {}})
from ray.rllib.algorithms.algorithm import Algorithm

def get_agent_cost_iteration(foldername, iteration, env):
    chkpt_dst = "/home/willem/policies/" + foldername + "/checkpoint_" + str(iteration).zfill(6)
    # https://github.com/tensorflow/tensorflow/issues/18304
    my_policy = Algorithm.from_checkpoint(chkpt_dst)
    # make arrays
    width_array = 8
    obs_array = []
    action_array = []
    energy_cost_array = []
    power_from_battery_array = []
    obs_array_descaled = {}
    battery_percentage_vec = []
    # reset the environment
    obs, info = env.reset()
    obs_dict=obs
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array = np.empty((0, width_array))
    while (env.terminated == False):
        # determine action
        action = my_policy.compute_single_action(obs_dict, explore=False)[0]
        # step the environment
        output = env.step([action])
        # read out observatrions
        obs = output[0]
        obs_dict = obs
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
    energy_cost = sum(energy_cost_array)
    return total_grid_cost, energy_cost, power_from_battery_array, action_array

agent_cost_arr = []
energy_cost_arr = []
foldername='special_reward_chkpt_test'
k=31
action_mat = np.empty((0,95))
pfb_mat = np.empty((0,95))
for i in range(k):
    agent_cost, energy_cost, pfb_arr, action_arr = get_agent_cost_iteration(foldername, i+1, env)
    agent_cost_arr = np.append(agent_cost_arr, agent_cost)
    energy_cost_arr = np.append(energy_cost_arr, energy_cost)
    pfb_mat = np.vstack((pfb_mat, pfb_arr))
    action_mat = np.vstack((action_mat, action_arr))
    print(i)


plt.figure()
l = 0
colors=plt.cm.jet(np.linspace(0,1,k-l))
for i in range(l,k):
    plt.plot(action_mat[i,:], color=colors[i-l], label=str(i))
plt.title('agent input')
plt.legend()


plt.figure()
l = 0
colors=plt.cm.jet(np.linspace(0,1,k-l))
for i in range(l,k):
    plt.plot(pfb_mat[i,:], color=colors[i-l], label=str(i))
plt.title('agent input converted to pfb by env')
plt.legend()