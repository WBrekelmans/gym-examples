# https://github.com/ray-project/ray/issues/7983

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# inputs
chkpt_dst = "/home/willem/policies/week_with_gen_20/checkpoint_000020/policies/default_policy"

# register the environment
from ray.tune.registry import register_env
from gym_examples.envs.EMS import EnergyManagementEnv
select_env = "gym_examples/EMS-v0"
register_env(select_env, lambda config: EnergyManagementEnv())

# load chkpt
from ray.rllib.algorithms.algorithm import Policy
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# https://github.com/tensorflow/tensorflow/issues/18304
my_policy = Policy.from_checkpoint(chkpt_dst)

# make the environment
import gymnasium as gym
import gym_examples
from gymnasium.wrappers import FlattenObservation
env = gym.make('gym_examples/EMS-v0')
env_f = FlattenObservation(env)

# reset environment and get first obs
import numpy as np
obs, info = env.reset()
obs = np.fromiter(obs.values(),dtype='float')
action = my_policy.compute_single_action(obs)[0]

i = 0
width_array = 10
length_array = 96*7

obs_array = np.zeros(shape=(length_array, width_array))
action_array = np.zeros(shape=(length_array, 2))
energy_cost_array = np.zeros(shape=(length_array, 1))
battery_percentage_vec = []

while not env.terminated:
    # get output
    output = env.step(action)
    # read out observatrions
    obs = output[0]
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array[i,:] = obs
    #
    action = my_policy.compute_single_action(obs)[0]
    action_array[i,:] = action
    energy_cost_array[i] = output[4]['energy_cost']
    print(i)
    i += 1

plt.figure()
plt.plot(obs_array[:,3], label='usage')
plt.plot(obs_array[:,4], label='day ahead price')
plt.plot(obs_array[:,5], label='solar_generation')
plt.plot(obs_array[:,6], label='soc_battery')
plt.plot(obs_array[:,7], label='power from grid')
plt.plot(obs_array[:,8], label='power from battery')
plt.plot(obs_array[:,9], label='power from gen')
plt.legend()
plt.show()

# descale values
obs_array_descaled = obs_array
obs_array_descaled[:,0] = env.descale_value(obs_array_descaled[:,0], env.range_dict['day_year'][0], env.range_dict['day_year'][1])
obs_array_descaled[:,1] = env.descale_value(obs_array_descaled[:,1], env.range_dict['day_week'][0], env.range_dict['day_week'][1])
obs_array_descaled[:,2] = env.descale_value(obs_array_descaled[:,2], env.range_dict['quarter_day'][0], env.range_dict['quarter_day'][1])
obs_array_descaled[:,3] = env.descale_value(obs_array_descaled[:,3], env.range_dict['usage'][0], env.range_dict['usage'][1])
obs_array_descaled[:,4] = env.descale_value(obs_array_descaled[:,4], env.range_dict['day_ahead_price'][0], env.range_dict['day_ahead_price'][1])
obs_array_descaled[:,5] = env.descale_value(obs_array_descaled[:,5], env.range_dict['solar_generation'][0], env.range_dict['solar_generation'][1])
obs_array_descaled[:,6] = env.descale_value(obs_array_descaled[:,6], env.range_dict['soc_battery'][0], env.range_dict['soc_battery'][1])
obs_array_descaled[:,7] = env.descale_value(obs_array_descaled[:,7], env.range_dict['power_from_grid'][0], env.range_dict['power_from_grid'][1])
obs_array_descaled[:,8] = env.descale_value(obs_array_descaled[:,8], env.range_dict['power_from_battery'][0], env.range_dict['power_from_battery'][1])
obs_array_descaled[:,9] = env.descale_value(obs_array_descaled[:,9], env.range_dict['power_from_generator'][0], env.range_dict['power_from_generator'][1])

power_from_generator = obs_array_descaled[:,9]
power_from_grid = obs_array_descaled[:,7]
day_ahead_price = obs_array_descaled[:,4]
cost_generator = 0.60
cost_test = power_from_generator * cost_generator * 0.25 + 0.25 * power_from_grid*day_ahead_price

total_quarter = (obs_array_descaled[:,0] - 1) * 96 + obs_array_descaled[:,2]



plt.figure()
plt.plot(obs_array_descaled[:,3], label='usage')
plt.plot(obs_array_descaled[:,4], label='day ahead price')
plt.plot(obs_array_descaled[:,5], label='solar_generation')
plt.plot(obs_array_descaled[:,6], label='soc_battery')
plt.plot(obs_array_descaled[:,7], label='power from grid')
plt.plot(obs_array_descaled[:,8], label='power from battery')
plt.plot(obs_array_descaled[:,9], label='power from gen')
plt.legend()
plt.show()


energy_balance = obs_array_descaled[:,3] - obs_array_descaled[:,5] - obs_array_descaled[:,8] - obs_array_descaled[:,9] - obs_array_descaled[:,7]

fig,axs = plt.subplots(3,3)
axs = axs.flatten()
k = 96*6
axs[0].plot(obs_array_descaled[0:k,3], label='usage')
axs[0].set_title('usage')
axs[1].plot(obs_array_descaled[0:k,4], label='day ahead price')
axs[1].set_title('day ahead price')
axs[2].plot(obs_array_descaled[0:k,5], label='solar_generation')
axs[2].set_title('solar_generation')
axs[3].plot(obs_array_descaled[0:k,6], label='soc_battery')
axs[3].set_title('soc_battery')
axs[4].plot(obs_array_descaled[0:k,7], label='power from grid')
axs[4].set_title('power from grid')
axs[5].plot(obs_array_descaled[0:k,8], label='power from battery')
axs[5].set_title('power from battery')
axs[6].plot(obs_array_descaled[0:k,9], label='power from gen')
axs[6].set_title('power from gen')
axs[7].plot(energy_balance[0:k], label='energy balance')
axs[7].set_title('energy balance')
axs[8].plot(energy_cost_array[0:k], label='energy cost')
axs[8].set_title('energy cost')
plt.legend()

# energy cost
total_gen_cost = sum(obs_array_descaled[0:k,9])*0.25*0.60
total_grid_cost = sum(obs_array_descaled[0:k,7]*obs_array_descaled[0:k,4])*0.25
print('total gen cost: ' + str(total_gen_cost))
print('total grid cost: ' + str(total_grid_cost))
print(' total cost: ' + str(total_gen_cost + total_grid_cost))

# compare to baseline
baseline = np.sum(obs_array_descaled[:,3] * obs_array_descaled[:,4])
cost_ratio = (total_gen_cost + total_grid_cost) / baseline
print('fraction cost compared to baseline: ' + str(cost_ratio))