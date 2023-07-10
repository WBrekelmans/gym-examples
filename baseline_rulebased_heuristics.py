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

rule based heuristics:
if pv == load: supply load with pv gen
if pv > load: charge battery
    charge battery till max capacity, if max capacity:
        sell back to grid
if pv < load: discharge battery
    discharge battery till empty, iff empty:
        get from grid
"""

# inputs
start_index = 0
stop_index = 10
data_path = '/home/willem/data_analysis/'
filename = 'df_env.pkl'
cost_gen = 0.06  # euro per w
charge_rate = 4e5/4


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

def charge_battery(SOC, energy, charge_rate):
    if energy>charge_rate:
        for_battery = charge_rate
        excess = energy

# rule based heuristics
def rulebased_heuristics(input_data, charge_rate):
    L = stop_index - start_index
    cost = np.zeros([1,L])
    SOC = np.zeros([1,L])
    for i in range(L):
        if input_data['solar_generation'] == input_data['usage']:
            cost[i] = 0
        if input_data['solar_generation'] > input_data['usage']:

