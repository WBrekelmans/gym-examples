"""
to minimize: energy cost
vars to optimize: [generator actions], [battery actions]
input data: [day ahead price], [solar], [usage]
[energy cost] = [generator cost] + [grid cost]
generator cost = [generator actions] * gen price
[grid cost] = [power from grid] * [day ahead price]
[power from grid] = [usage] - [generator actions] - [battery actions] - [solar]
[soc] = [soc@t-1] - 0.25[battery actions]
constraints:
0 < [soc] < battery capacity
-charge rate < [battery actions] < charge rate
"""

# import libraries
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import NonlinearConstraint
import pandas as pd

# inputs
start_index = 0
stop_index = 10
data_path = '/home/willem/data_analysis/'
filename = 'df_env.pkl'
cost_gen = 0.06  # euro per w


# load data
def load_data(start_index, stop_index, data_path, filename):
    df_env = pd.read_pickle(data_path + filename)
    df_env['usage'] = df_env['usage'] * 4  # from wh to w
    df_env['solar_generation'] = df_env['solar_generation'] * 4  # from wh to w
    df_env['day_ahead_price'] = df_env['day_ahead_price']  # price per kwh
    input_data = {
        'usage': df_env['usage'][start_index:stop_index],
        'solar_generation': df_env['solar_generation'][start_index:stop_index],
        'day_ahead_price': df_env['day_ahead_price'][start_index:stop_index]
    }
    return input_data


input_data = load_data(start_index, stop_index, data_path, filename)


# main function to optimize

def target_function(charge_level):
    generator_cost = generator_actions * cost_gen
    power_from_grid = input_data['usage'] - generator_actions - battery_actions - input_data['solar_generation']
    grid_cost = power_from_grid * input_data['day_ahead_price']
    energy_cost = generator_cost + grid_cost
    return energy_cost


def constraint_function(generator_actions, battery_actions):
    SOC = np.zeros([1, stop_index - start_index])
    SOC[0] = 0
    for i in range(1, stop_index - start_index-1):
        SOC[i] = SOC[i - 1] - 0.25 * battery_actions[i]
    return SOC


lower_limit = np.ones([1, stop_index - start_index]) * 0
upper_limit = np.ones([1, stop_index - start_index]) * 4e5
constraint = NonlinearConstraint(constraint_function, 0, 4e5)

# make pbounds
pbounds = {}
for i in range(stop_index-start_index):

pbounds = {'generator_actions': (0, 243200), 'battery_actions': (-4e5, 4e5)}

optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=0,
    verbose=0
)

optimizer.maximize(
    init_points=2,
    n_iter=100
)

print(f'the best solution with no constraints is {optimizer.max}')
