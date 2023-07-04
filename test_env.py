import gymnasium as gym
import gym_examples
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# make the environment
env = gym.make('gym_examples/GridWorld-v0')
env = gym.make('gym_examples/EMS-v0')
env = gym.make('gym_examples/EMS_no_gen-v0')
from gym_examples.envs.EMS import EnergyManagementEnv
env = EnergyManagementEnv()

# reset environment and get first obs
import numpy as np
obs, info = env.reset()
obs = np.fromiter(obs.values(),dtype='float')
action = [1,-1]

i = 0
width_array = 10
length_array = 95

obs_array = np.zeros(shape=(length_array, width_array))
action_array = np.zeros(shape=(length_array, 2))
energy_cost_array = np.zeros(shape=(length_array, 1))
energy_from_battery_PFB_array = np.zeros(shape=(length_array, 1))
energy_from_battery_SOC_array = np.zeros(shape=(length_array, 1))

battery_percentage_vec = []

while not env.terminated:
    if i < 6:
        action = [1, 0]
    else:
        action = [0,0]
    # get output
    output = env.step(action)
    # read out observatrions
    obs = output[0]
    obs = np.fromiter(obs.values(), dtype='float')
    obs_array[i,:] = obs
    energy_cost_array[i] = output[4]['energy_cost']
    print(i)
    i += 1

total_quarter = (obs_array[:,0] - 1) * 96 + obs_array[:,2]

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

# energy balance
# self._usage - self._solar_generation - self._power_from_battery - self._power_from_generator - self._power_from_grid
energy_balance = obs_array[:,3] - obs_array[:,5] - obs_array[:,8] - obs_array[:,9] - obs_array[:,7]
plt.plot(energy_balance)

# descale
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
energy_balance_descaled = obs_array_descaled[:,3] - obs_array_descaled[:,5] - obs_array_descaled[:,8] - obs_array_descaled[:,9] - obs_array_descaled[:,7]


# plot battery and SOC
f,ax = plt.subplots()
ax.plot(obs_array_descaled[:,6], label='SOC battery kwh')
ax.plot(obs_array_descaled[:,8], label='power from battery kw')
ax.legend()
ax.set_xticks(np.arange(1,96,1))
ax.set_yticks(np.arange(0,max(max(obs_array_descaled[:,6]), max(obs_array_descaled[:,8])), 10000))
ax.grid(which='both')


fig,axs = plt.subplots(3,3)
axs = axs.flatten()
k = -1
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
axs[7].plot(energy_balance_descaled[0:k], label='energy balance')
axs[7].set_title('energy balance')
axs[8].plot(energy_cost_array[0:k], label='energy cost')
axs[8].set_title('energy cost')
plt.legend()

plt.figure()
energy_balance = obs_array_descaled[:,3] - obs_array_descaled[:,5] - obs_array_descaled[:,8] - obs_array_descaled[:,9] - obs_array_descaled[:,7]
plt.plot(energy_balance)

# test energy in and out of battery
energy_from_battery_PFB = -1*sum(obs_array_descaled[:,8])/4
energy_from_battery_SOC = obs_array_descaled[:,6][-1] - obs_array_descaled[:,6][0]

# test energy cost
cost_generator = 0.25 * env.generator_kwh_price * obs_array_descaled[0:k,9]
cost_grid = 0.25 * env._day_ahead_price * obs_array_descaled[0:k,7]
energy_cost = cost_grid + cost_grid
fig,axs = plt.subplots(3,1)
axs[0].plot(cost_generator, label='cost_generator')
axs[1].plot(cost_grid, label='cost_grid')
axs[2].plot(energy_cost, label='energy_cost')
