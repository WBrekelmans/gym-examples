# https://github.com/ray-project/ray/issues/7983
# max power from battery gedeeld door vier

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from gym_examples.envs.EMS_no_gen_V2 import EnergyManagementEnv_no_gen_V2
from ray.rllib.algorithms.algorithm import Policy
import tensorflow as tf
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym

matplotlib.use('TkAgg')

# inputs
chkpt_dst = "/home/willem/policies/special_reward/checkpoint_000033/policies/default_policy"

# register and make the environment
select_env = "gym_examples/EMS_no_gen-v2"
register_env(select_env, lambda config: EnergyManagementEnv_no_gen_V2())
env = gym.make(select_env)

# load chkpt
tf.compat.v1.enable_eager_execution()
# https://github.com/tensorflow/tensorflow/issues/18304
my_policy = Policy.from_checkpoint(chkpt_dst)

# array
i = 0
width_array = 8

obs_array = []
action_array = []
energy_cost_array = []
power_from_battery_array = []
battery_percentage_vec = []

# reset the environment
obs, info = env.reset()
obs = np.fromiter(obs.values(), dtype='float')
obs_array = np.empty((0,width_array))
obs_array = np.vstack((obs_array, obs))

i=1
while (env.terminated==False):
    # determine action
    action = my_policy.compute_single_action([obs], explore=False)[0]
    # step the environment
    output = env.step(action)
    # read out observatrions
    obs = output[0]
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array = np.vstack((obs_array,obs))
    action_array = np.append(action_array, action)
    energy_cost_array = np.append(energy_cost_array, output[1])
    power_from_battery_array = np.append(power_from_battery_array, output[4]['power_from_battery'])
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
obs_array_descaled = {}
obs_array_descaled['day_year'] = env.descale_value(obs_array[:,0], env.range_dict['day_year'][0], env.range_dict['day_year'][1])
obs_array_descaled['day_week'] = env.descale_value(obs_array[:,1], env.range_dict['day_week'][0], env.range_dict['day_week'][1])
obs_array_descaled['quarter_day'] = env.descale_value(obs_array[:,2], env.range_dict['quarter_day'][0], env.range_dict['quarter_day'][1])
obs_array_descaled['usage'] = env.descale_value(obs_array[:,3], env.range_dict['usage'][0], env.range_dict['usage'][1])
obs_array_descaled['day_ahead_price'] = env.descale_value(obs_array[:,4], env.range_dict['day_ahead_price'][0], env.range_dict['day_ahead_price'][1])
obs_array_descaled['solar_generation'] = env.descale_value(obs_array[:,5], env.range_dict['solar_generation'][0], env.range_dict['solar_generation'][1])
obs_array_descaled['soc_battery'] = env.descale_value(obs_array[:,6], env.range_dict['soc_battery'][0], env.range_dict['soc_battery'][1])
obs_array_descaled['power_from_grid'] = env.descale_value(obs_array[:,7], env.range_dict['power_from_grid'][0], env.range_dict['power_from_grid'][1])
obs_array_descaled['power_from_battery'] = env.descale_value(power_from_battery_array, env.range_power_from_battery[0], env.range_power_from_battery[1])

# plot descaled
fig,axs=plt.subplots(2,1,sharex=True)
axs=axs.flatten()
axs[0].plot(obs_array_descaled['usage'], label='usage')
axs[0].plot(obs_array_descaled['day_ahead_price'], label='day ahead price')
axs[0].plot(obs_array_descaled['solar_generation'], label='solar_generation')
axs[0].plot(obs_array_descaled['soc_battery'], label='soc_battery')
axs[0].plot(obs_array_descaled['power_from_grid'], label='power from grid')
axs[0].plot(obs_array_descaled['power_from_battery'], label='power from battery')
axs[0].legend()
axs[1].plot(obs_array_descaled['day_ahead_price'], label='day ahead price')
plt.show()

# subplots descaled
fig,axs = plt.subplots(4,2)
axs = axs.flatten()
axs[0].plot(obs_array_descaled['usage'], label='usage')
axs[0].set_title('usage')
axs[1].plot(obs_array_descaled['day_ahead_price'], label='day ahead price')
axs[1].set_title('day ahead price')
axs[2].plot(obs_array_descaled['solar_generation'], label='solar_generation')
axs[2].set_title('solar_generation')
axs[3].plot(obs_array_descaled['soc_battery'], label='soc_battery')
axs[3].set_title('soc_battery')
axs[4].plot(obs_array_descaled['power_from_grid'], label='power from grid')
axs[4].set_title('power from grid')
axs[5].plot(obs_array_descaled['power_from_battery'], label='power from battery')
axs[5].set_title('power from battery')
axs[6].plot(energy_cost_array, label='energy cost agent')
axs[6].set_title('energy cost agent')
plt.legend()

# energy cost
grid_cost = 0.25*obs_array_descaled['power_from_grid']*obs_array_descaled['day_ahead_price']
total_grid_cost = sum(grid_cost)
print('total grid cost: ' + str(total_grid_cost))
print(' total cost: ' + str(total_grid_cost))

# compare to baseline
baseline = 0.25*obs_array_descaled['usage']*obs_array_descaled['day_ahead_price']
total_baseline = sum(baseline)
cost_ratio = (total_grid_cost) / total_baseline
print('fraction cost compared to baseline: ' + str(cost_ratio))

# solar baseline
energy_usage_solar = obs_array_descaled['usage'] - obs_array_descaled['solar_generation']
energy_usage_solar[energy_usage_solar<0] = 0
solar_baseline = energy_usage_solar*obs_array_descaled['day_ahead_price']*0.25
total_solar_baseline = sum(solar_baseline)
cost_ratio_solar = (total_grid_cost) / total_solar_baseline
print('fraction cost compared to solar baseline: ' + str(cost_ratio_solar))

# solar with sell-back baseline
energy_usage_solar = obs_array_descaled['usage'] - obs_array_descaled['solar_generation']
solar_baseline_sellback = energy_usage_solar*obs_array_descaled['day_ahead_price']*0.25
total_solar_baseline_sellback = sum(solar_baseline_sellback)
cost_ratio_solar_sellback = (total_grid_cost) / total_solar_baseline_sellback
print('fraction cost compared to solar baseline sell-back: ' + str(cost_ratio_solar_sellback))

# baseline and actual energy cost
f,ax = plt.subplots()
ax.plot(grid_cost, label='agent cost')
ax.plot(baseline, label='baseline')
ax.plot(len(solar_baseline), label='solar')
ax.plot(solar_baseline_sellback, label='solar sellback')
ax.legend()



# test PFG when no battery
# scale
# DDPG
# make it one day

plt.figure()
plt.plot(action_array, label='actions by agent',linestyle="-", marker="o")
plt.plot(power_from_battery_array, label='power from battery',linestyle="-", marker="o")
plt.plot(obs_array[:,6], label='soc',linestyle="-", marker="o")
plt.axhline(y=-1, color='r', linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.legend()