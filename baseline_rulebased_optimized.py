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
# Inputs
start_index = 101*24*4
stop_index = start_index + 24*4*1
data_path = '/home/willem/data_analysis/'
filename = 'df_env.pkl'
cost_gen = 0.6  # euro per w
battery_capacity = 4e5  # in wh
charge_rate = battery_capacity / 4

# import libraries
import numpy as np
import pandas as pd


# load data
def load_data(start_index, stop_index, data_path, filename):
    df_env = pd.read_pickle(data_path + filename)
    df_env['usage'] = df_env['usage'] * 4  # from wh to w
    df_env['solar_generation'] = df_env['solar_generation'] * 4  # from wh to w
    df_env['day_ahead_price'] = df_env['day_ahead_price']  # price per kwh
    input_data = {
        'usage': df_env['usage'][start_index:stop_index].values,
        'solar_generation': df_env['solar_generation'][start_index:stop_index].values,
        'day_ahead_price': df_env['day_ahead_price'][start_index:stop_index].values
    }
    return input_data


input_data = load_data(start_index, stop_index, data_path, filename)
solar_data = input_data['solar_generation']
load_data = input_data['usage']
grid_price_data = input_data['day_ahead_price']

# Create a Concrete Model
from pyomo.environ import *

model = ConcreteModel()

# Define the time index set
simulation_duration = stop_index - start_index
model.time = RangeSet(0, simulation_duration - 1)

# Parameters
model.S0 = Param(initialize=0, mutable=True)  # Wh SOC at t=0

# Variables
model.charge = Var(model.time, within=NonNegativeReals)  # Charging amount at each time step
model.discharge = Var(model.time, within=NonNegativeReals)  # Discharging amount at each time step
# model.generator_supply = Var(model.time, within=NonNegativeReals)  # Energy supply from the backup generator at each time step
model.SOC = Var(model.time, bounds=(0, battery_capacity))

# Objective function: Minimize energy costs
# model.objective = Objective(expr=sum(cost_gen*model.generator_supply[i] + grid_price_data[i] * (model.charge[i] + load_data[i] - model.discharge[i] - solar_data[i] - model.generator_supply[i])
#                                    for i in model.time), sense=minimize)

model.objective = Objective(
    expr=sum(0.25*grid_price_data[i] * (model.charge[i] + load_data[i] - model.discharge[i] - solar_data[i])
             for i in model.time), sense=minimize)

# Constraints charging discharging
model.C1_max_charge = ConstraintList()
model.C2_max_discharge = ConstraintList()
model.C3_power_balance = ConstraintList()
model.C4_SOC = ConstraintList()
model.C5_charge_or_discharge = ConstraintList()

for i in model.time:
    # Charging rate constraint
    model.C1_max_charge.add(model.charge[i] <= charge_rate)

    # Discharging rate constraint
    model.C2_max_discharge.add(model.discharge[i] <= charge_rate)

    # Constraints power balance
    model.C3_power_balance.add(load_data[i] + model.charge[i] == solar_data[i] + model.discharge[i] + (
                model.charge[i] + load_data[i] - model.discharge[i] - solar_data[i]))

def SOC_storage(model, i):
    if i == 0:
        return model.SOC[i] == model.S0 + 0.25*(model.charge[i] - model.discharge[i])
    else:
        return model.SOC[i] == model.SOC[i - 1] + 0.25*(model.charge[i] - model.discharge[i])

model.C4_SOC = Constraint(model.time,rule = SOC_storage)
# Additional constraint: Battery state of charge
# model.soc_constraint = Constraint(model.time, rule=lambda model, i: model.charge[i] - model.discharge[i]
#                                                             == model.charge[i-1] - model.discharge[i-1] if i > 1 else model.charge[i] - model.discharge[i] == 0)

# Solve the optimization problem
solver = SolverFactory('glpk')
solver.solve(model)

# Print the optimized charging schedule and energy supply from the grid
for i in model.time:
    print(f"Time: {i * 15 // 60}:{(i * 15) % 60:02d} |\
     Charge: {model.charge[i].value:.2f} |\
     Discharge: {model.discharge[i].value:.2f} |\
     Power from grid:  {(model.charge[i].value + load_data[i] - model.discharge[i].value - solar_data[i]):.2f}")

# Print total energy cost
total_cost = sum(0.25*grid_price_data[i] * (
            model.charge[i].value + load_data[i] - model.discharge[i].value - solar_data[i]) for i in model.time)
print(f"\nTotal Energy Cost: {total_cost:.2f}")

# to arrays
charge = []
discharge = []
SOC = []
power_from_grid = []
energy_cost = []
energy_balance = []

for i in model.time:
    charge = np.append(charge, model.charge[i].value)
    discharge = np.append(discharge, model.discharge[i].value)
    SOC = np.append(SOC, model.SOC[i].value)
    power_from_grid_curr = model.charge[i].value + load_data[i] - model.discharge[i].value - solar_data[i]
    power_from_grid = np.append(power_from_grid, power_from_grid_curr)
    energy_cost = np.append(energy_cost, power_from_grid_curr*grid_price_data[i])
    energy_balance_curr = load_data[i] - solar_data[i] - (model.discharge[i].value - model.charge[i].value) - power_from_grid_curr
    energy_balance = np.append(energy_balance, energy_balance_curr)

# plot
import matplotlib.pyplot as plt

fig,axs=plt.subplots(10,1,sharex=True)
axs=axs.flatten()
axs[0].plot(charge, label='charge')
axs[0].set_title('charge')

axs[1].plot(discharge, label='discharge')
axs[1].set_title('discharge')

axs[2].plot(discharge-charge, label='power_from_battery')
axs[2].set_title('power_from_battery')

axs[3].plot(SOC, label='SOC')
axs[3].set_title('SOC')

axs[4].plot(solar_data, label='solar')
axs[4].set_title('solar')

axs[5].plot(load_data, label='usage')
axs[5].set_title('usage')

axs[6].plot(power_from_grid, label='power from grid')
axs[6].set_title('power from grid')

axs[7].plot(grid_price_data, label='grid price')
axs[7].set_title('grid price')

axs[8].plot(energy_cost, label='energy cost')
axs[8].set_title('energy cost')

axs[9].plot(energy_balance, label='energy balance')
axs[9].set_title('energy balance')

plt.show()