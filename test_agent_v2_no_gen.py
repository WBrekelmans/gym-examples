# https://github.com/ray-project/ray/issues/7983
# max power from battery gedeeld door vier

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# inputs
chkpt_dst = "/home/willem/policies/v2_env_action_space_maybe_clipped/checkpoint_000001/policies/default_policy"

# register the environment
from ray.tune.registry import register_env
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
select_env = "gym_examples/EMS_no_gen-v2"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen_V2())

# load chkpt
from ray.rllib.algorithms.algorithm import Policy
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# https://github.com/tensorflow/tensorflow/issues/18304
my_policy = Policy.from_checkpoint(chkpt_dst)

# make the environmentw
import gymnasium as gym
import gym_examples
from gymnasium.wrappers import FlattenObservation
env = gym.make('gym_examples/EMS_no_gen-v2')
env_f = FlattenObservation(env)

# reset environment and get first obs
import numpy as np
obs, info = env.reset()
obs = np.fromiter(obs.values(),dtype='float')
action = my_policy.compute_single_action(obs)[0]

i = 0
width_array = 8
length_array = 96*7

obs_array = np.zeros(shape=(length_array, width_array))
action_array = np.zeros(shape=(length_array, 1))
energy_cost_array = np.zeros(shape=(length_array, 1))
power_from_battery_array = np.zeros(shape=(length_array, 1))
battery_percentage_vec = []

while not env.terminated:
    # get output
    output = env.step(action)
    # read out observatrions
    obs = output[0]
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array[i,:] = obs
    #
    action = my_policy.compute_single_action([obs], unquash_action=True)[0]
    action_array[i,:] = action
    energy_cost_array[i] = output[4]['energy_cost']
    power_from_battery_array[i] = output[4]['power_from_battery']
    print(i)
    i += 1

plt.figure()
plt.plot(obs_array[:,3], label='usage')
plt.plot(obs_array[:,4], label='day ahead price')
plt.plot(obs_array[:,5], label='solar_generation')
plt.plot(obs_array[:,6], label='soc_battery')
plt.plot(obs_array[:,7], label='power from grid')
plt.plot(power_from_battery_array, label='power from battery')
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
power_from_battery_array_descaled = env.descale_value(power_from_battery_array, env.range_power_from_battery[0], env.range_power_from_battery[1])

power_from_grid = obs_array_descaled[:,7]
day_ahead_price = obs_array_descaled[:,4]
cost_test = 0.25 * power_from_grid*day_ahead_price

total_quarter = (obs_array_descaled[:,0] - 1) * 96 + obs_array_descaled[:,2]

plt.figure()
plt.plot(obs_array_descaled[:,3], label='usage')
plt.plot(obs_array_descaled[:,4], label='day ahead price')
plt.plot(obs_array_descaled[:,5], label='solar_generation')
plt.plot(obs_array_descaled[:,6], label='soc_battery')
plt.plot(obs_array_descaled[:,7], label='power from grid')
plt.plot(power_from_battery_array_descaled, label='power from battery')
plt.legend()
plt.show()


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
axs[5].plot(power_from_battery_array_descaled, label='power from battery')
axs[5].set_title('power from battery')
#axs[7].plot(energy_balance[0:k], label='energy balance')
#axs[7].set_title('energy balance')
axs[8].plot(energy_cost_array[0:k], label='energy cost')
axs[8].set_title('energy cost')
plt.legend()

# energy cost
grid_cost = obs_array_descaled[:,7]*obs_array_descaled[:,4]*0.25
total_grid_cost = sum(grid_cost)
print('total grid cost: ' + str(total_grid_cost))
print(' total cost: ' + str(total_grid_cost))

# compare to baseline
baseline = obs_array_descaled[:,3] * obs_array_descaled[:,4]*0.25
total_baseline = sum(baseline)
cost_ratio = (total_grid_cost) / total_baseline
print('fraction cost compared to baseline: ' + str(cost_ratio))

# solar baseline
energy_usage_solar = obs_array_descaled[:,3] - obs_array_descaled[:,5]
energy_usage_solar[energy_usage_solar<0] = 0
solar_baseline = energy_usage_solar*obs_array_descaled[:,4]*0.25
total_solar_baseline = sum(solar_baseline)
cost_ratio_solar = (total_grid_cost) / total_solar_baseline
print('fraction cost compared to solar baseline: ' + str(cost_ratio_solar))

# solar with sell-back baseline
energy_usage_solar = obs_array_descaled[:,3] - obs_array_descaled[:,5]
solar_baseline_sellback = energy_usage_solar*obs_array_descaled[:,4]*0.25
total_solar_baseline_sellback = sum(solar_baseline_sellback)
cost_ratio_solar_sellback = (total_grid_cost) / total_solar_baseline_sellback
print('fraction cost compared to solar baseline sell-back: ' + str(cost_ratio_solar_sellback))

# baseline and actual energy cost
f,ax = plt.subplots()
ax.plot(grid_cost, label='agent cost')
ax.plot(baseline, label='baseline')
ax.plot(solar_baseline, label='solar')
ax.plot(solar_baseline_sellback, label='solar sellback')
ax.legend()
