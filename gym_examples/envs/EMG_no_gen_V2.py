import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import scipy.integrate as integrate

class EnergyManagementEnv_no_gen_V2(gym.Env):
    def __init__(self):
        self.NUMBER_OF_DAYS = 7
        self.FIX_SOC = True
        self.FIX_TIME = True
        self.battery_capacity = 4e5  # kwh
        self.charge_rate_battery = self.battery_capacity  # full in one hour, in kw/h

        self.range_dict_disc = {"day_year": [1, 365], "day_week": [1, 7], "quarter_day": [1, 96]}
        self.range_dict_disc_r = {"day_year": [-1, 1], "day_week": [-1, 1], "quarter_day": [-1, 1]}
        self.range_dict_cont = {"usage": [75000.0 * 4, 249000.0 * 4],
                                "day_ahead_price": [-0.00022236, 0.000871],
                                "solar_generation": [0.0, 93750.0 * 4],
                                "soc_battery": [0.0, self.battery_capacity],
                                "power_from_grid": [0.0, 0.0],
                                "power_to_grid": [0.0, 0.0],
                                "power_from_battery": [-self.charge_rate_battery, self.charge_rate_battery]}
        self.range_dict_cont_r = {"usage": [-1, 1], "day_ahead_price": [-1, 1],
                                  "solar_generation": [-1, 1], "soc_battery": [-1, 1],
                                  "power_from_grid": [-1, 1],
                                  "power_from_battery": [-1, 1]}
        self.terminated = False
        self.truncated = False
        self.prob_obs = ["soc_battery", "power_from_battery"]

        self.discrete_states = {}
        self.discrete_states = {
            disc_value: spaces.Box(low=self.range_dict_disc_r[disc_value][0],
                                   high=self.range_dict_disc_r[disc_value][1],
                                   shape=(1,), dtype=int) for disc_value in self.range_dict_disc_r.keys()}
        self.cont_states = {
            cont_value: spaces.Box(low=self.range_dict_cont_r[cont_value][0],
                                   high=self.range_dict_cont_r[cont_value][1],
                                   shape=(1,), dtype=np.float64) for cont_value in self.range_dict_cont_r.keys()}
        self.states = self.discrete_states
        self.states.update(self.cont_states)
        self.observation_space = spaces.Dict(self.states)

        self.range_dict = self.range_dict_disc
        self.range_dict.update(self.range_dict_cont)

        self.range_dict_r = self.range_dict_disc_r
        self.range_dict_r.update(self.range_dict_cont_r)

        self.power_from_grid_max = self.range_dict['usage'][1] - self.range_dict['solar_generation'][0] - self.range_dict['power_from_battery'][0]
        self.power_from_grid_min = self.range_dict['usage'][0] - self.range_dict['solar_generation'][1] - self.range_dict['power_from_battery'][1]
        self.range_dict['power_from_grid'][0] = self.power_from_grid_min
        self.range_dict['power_from_grid'][1] = self.power_from_grid_max

        # power from battery, power from generator
        self.action_space = spaces.Box(low=np.array([self.range_dict_r["power_from_battery"][0]]),
                                       high=np.array([self.range_dict_r["power_from_battery"][1]]),
                                       shape=(1,), dtype=np.float64)

        self.data_path = '/home/willem/data_analysis/'
        self.filename = 'df_env.pkl'
        self.df_env = pd.read_pickle(self.data_path + self.filename)
        self.df_env['usage'] = self.df_env['usage'] * 4  # make usage kw
        self.df_env['solar_generation'] = self.df_env['solar_generation'] * 4  # make solar kw

    def scale_eq(self, value, min_meas, max_meas, min_des, max_des):
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        return ((value - min_meas) / (max_meas - min_meas)) * (max_des - min_des) + min_des

    def scale_value(self, value, min_meas, max_meas):
        return self.scale_eq(value, min_meas, max_meas, -1, 1)

    def descale_value(self, value, min_des, max_des):
        return self.scale_eq(value, -1, 1, min_des, max_des)

    def _get_obs(self):
        return {
            "day_year": np.array([self._day_year]),
            "day_week": np.array([self._day_week]),
            "quarter_day": np.array([self._quarter_day]),
            "usage": np.array([self._usage]),
            "day_ahead_price": np.array([self._day_ahead_price]),
            "solar_generation": np.array([self._solar_generation]),
            "soc_battery": np.array([self._soc_battery]),
            "power_from_grid": np.array([self._power_from_grid]),
            "power_from_battery": np.array([self._power_from_battery])

        }

    def get_energy_cost(self):
        power_from_grid = self.descale_value(self._power_from_grid, self.range_dict['power_from_grid'][0],
                                             self.range_dict['power_from_grid'][1])
        day_ahead_price = self.descale_value(self._day_ahead_price, self.range_dict['day_ahead_price'][0],
                                             self.range_dict['day_ahead_price'][1])
        cost_grid = 0.25 * day_ahead_price * power_from_grid
        energy_cost = cost_grid
        return energy_cost

    def _get_info(self):
        return {
            "battery_percentage": (self._soc_battery / self.battery_capacity) * 100,  # battery charge percentage
            "energy_cost": self.get_energy_cost(),
        }

    def update_determined_obs(self):
        day_year = round(self.descale_value(self._day_year, self.range_dict['day_year'][0],
                                            self.range_dict['day_year'][1]))
        quarter_day = round(self.descale_value(self._quarter_day, self.range_dict['quarter_day'][0],
                                               self.range_dict['quarter_day'][1]))
        total_quarter = round((day_year - 1) * 96 + quarter_day)
        print('day year is ' + str(day_year))
        print('quarter day is ' + str(quarter_day))
        print('total quarter is ' + str(total_quarter))
        usage = self.df_env.usage[total_quarter].astype(np.float64)
        day_ahead_price = self.df_env.day_ahead_price[total_quarter].astype(np.float64)
        solar_generation = self.df_env.solar_generation[total_quarter].astype(np.float64)
        self._usage = self.scale_value(usage, self.range_dict['usage'][0], self.range_dict['usage'][1])
        self._day_ahead_price = self.scale_value(day_ahead_price, self.range_dict['day_ahead_price'][0],
                                                 self.range_dict['day_ahead_price'][1])
        self._solar_generation = self.scale_value(solar_generation, self.range_dict['solar_generation'][0],
                                                  self.range_dict['solar_generation'][1])
        return

    def check_in_space(self):
        for key in self.states.keys():
            if self._get_obs()[key] < self.range_dict[key][0] or self._get_obs()[key] > self.range_dict[key][1]:
                print('OUT OF BOUNDS:')
                print(key)
                print('range:')
                print(self.range_dict[key])
                print('value:')
                print(self._get_obs()[key])
        return

    def reset_probabilistic_time_obs(self):
        day_year = self.np_random.integers(low=1, high=365, dtype=int)
        quarter_day = self.np_random.integers(low=1, high=96, dtype=int)
        if self.FIX_TIME:
            day_year = 100
            quarter_day = 1
        day_week = day_year % 7 + 1
        self._start_quarter = quarter_day
        self._day_year = self.scale_value(day_year, self.range_dict['day_year'][0],
                                          self.range_dict['day_year'][1])
        self._quarter_day = self.scale_value(quarter_day, self.range_dict['quarter_day'][0],
                                             self.range_dict['quarter_day'][1])
        self._day_week = self.scale_value(day_week, self.range_dict['day_week'][0],
                                          self.range_dict['day_week'][1])
        return

    def reset_probabilistic_obs(self):
        for prob_obs in self.prob_obs:
            value_to_set = np.float64(self.np_random.uniform(low=-1,
                                                             high=1))
            setattr(self, '_' + prob_obs, value_to_set)
        if self.FIX_SOC:
            self._soc_battery = 0
            self._power_from_battery = 0
        self.next_SOC = self._soc_battery
        return

    def reset_power_from_grid(self):
        usage = self.descale_value(self._usage, self.range_dict['usage'][0], self.range_dict['usage'][1])
        solar_generation = self.descale_value(self._solar_generation, self.range_dict['solar_generation'][0],
                                              self.range_dict['solar_generation'][1])
        power_from_battery = self.descale_value(self._power_from_battery, self.range_dict['power_from_battery'][0],
                                                self.range_dict['power_from_battery'][1])
        power_from_grid = usage - solar_generation - power_from_battery
        self._power_from_grid = self.scale_value(power_from_grid, self.range_dict['power_from_grid'][0],
                                                 self.range_dict['power_from_grid'][1])
        return

    def test_energy_balance(self):
        usage = self.descale_value(self._usage, self.range_dict['usage'][0], self.range_dict['usage'][1])
        solar_generation = self.descale_value(self._solar_generation, self.range_dict['solar_generation'][0],
                                              self.range_dict['solar_generation'][1])
        power_from_battery = self.descale_value(self._power_from_battery, self.range_dict['power_from_battery'][0],
                                                self.range_dict['power_from_battery'][1])
        power_from_grid = self.descale_value(self._power_from_grid, self.range_dict['power_from_grid'][0],
                                             self.range_dict['power_from_grid'][1])
        balance = usage - solar_generation - power_from_battery - power_from_grid
        extensive_print = False
        if extensive_print:
            print('usage: ' + str(usage))
            print('solar generation:' + str(solar_generation))
            print('power_from_battery:' + str(power_from_battery))
            print('power_from_grid:' + str(power_from_grid))
        print('balance ' + str(balance))
        return

    def reset(self, seed=None, options=None):
        self._quarter_counter = 1
        self.terminated = False
        super().reset(seed=seed)
        self.reset_probabilistic_time_obs()
        self.update_determined_obs()
        self.reset_probabilistic_obs()
        self.reset_power_from_grid()
        self.test_energy_balance()
        obs = self._get_obs()
        info = self._get_info()
        print('beer')
        return obs, info

    def step_proceed_quarter(self):
        quarter_day = round(self.descale_value(self._quarter_day, self.range_dict['quarter_day'][0],
                                               self.range_dict['quarter_day'][1]))
        quarter_day = quarter_day + 1
        if quarter_day < 97:
            self._quarter_day = self.scale_value(quarter_day, self.range_dict['quarter_day'][0],
                                                 self.range_dict['quarter_day'][1])
        else:
            quarter_day = 1
            self._quarter_day = self.scale_value(quarter_day, self.range_dict['quarter_day'][0],
                                                 self.range_dict['quarter_day'][1])
            day_week = round(self.descale_value(self._day_week, self.range_dict['day_week'][0],
                                                self.range_dict['day_week'][1]))
            day_week += 1
            if day_week < 8:
                self._day_week = self.scale_value(day_week, self.range_dict['day_week'][0],
                                                  self.range_dict['day_week'][1])
            else:
                day_week = 1
                self._day_week = self.scale_value(day_week, self.range_dict['day_week'][0],
                                                  self.range_dict['day_week'][1])
            day_year = round(self.descale_value(self._day_year, self.range_dict['day_year'][0],
                                                self.range_dict['day_year'][1]))
            day_year += 1
            if day_year < 365:
                self._day_year = self.scale_value(day_year, self.range_dict['day_year'][0],
                                                  self.range_dict['day_year'][1])
            else:
                day_year = 1
                self._day_year = self.scale_value(day_year, self.range_dict['day_year'][0],
                                                  self.range_dict['day_year'][1])
        return

    def step_power_from_grid(self):
        usage = self.descale_value(self._usage, self.range_dict['usage'][0], self.range_dict['usage'][1])
        solar_generation = self.descale_value(self._solar_generation, self.range_dict['solar_generation'][0],
                                              self.range_dict['solar_generation'][1])
        power_from_battery = self.descale_value(self._power_from_battery, self.range_dict['power_from_battery'][0],
                                                self.range_dict['power_from_battery'][1])
        power_from_grid = usage - solar_generation - power_from_battery
        self._power_from_grid = self.scale_value(power_from_grid, self.range_dict['power_from_grid'][0],
                                                 self.range_dict['power_from_grid'][1])
        return

    def step_soc_battery_new(self):
        power_from_battery = self.descale_value(self._power_from_battery, self.range_dict['power_from_battery'][0],
                                                self.range_dict['power_from_battery'][1])
        old_soc_battery = self.descale_value(self._soc_battery, self.range_dict['soc_battery'][0],
                                             self.range_dict['soc_battery'][1])
        new_soc = old_soc_battery - power_from_battery * 0.25
        # kwh - kw*(0.25h)
        # new_soc = old_soc_battery - power_from_battery
        # battery is full
        if new_soc > self.range_dict['soc_battery'][1]:
            # pushing power to battery
            if power_from_battery < 0:
                # determine allowed power we can push to battery
                delta_soc = self.range_dict['soc_battery'][1] - old_soc_battery
                power_from_battery = -1 * delta_soc * 1 / 4
        # battery is empty
        if new_soc < self.range_dict['soc_battery'][0]:
            if power_from_battery > 0:
                # determine allowed power we can take from battery
                delta_soc = self.range_dict['soc_battery'][0] - old_soc_battery
                # taking power from battery
                power_from_battery = -1 * delta_soc * 1 / 4
        new_soc = np.clip(new_soc, self.range_dict["soc_battery"][0],
                          self.range_dict["soc_battery"][1])
        new_soc = self.scale_value(new_soc, self.range_dict['soc_battery'][0],
                                             self.range_dict['soc_battery'][1])
        self._soc_battery = new_soc
        self._power_from_battery = self.scale_value(power_from_battery, self.range_dict['power_from_battery'][0],
                                                    self.range_dict['power_from_battery'][1])
        return

    def punish_soc(self):
        if self.soc_battery_new > self.range_dict['soc_battery'][1] * 0.90 or self.soc_battery_new < \
                self.range_dict['soc_battery'][1] * 0.10:
            reward = -10e6
        else:
            reward = 0
        return reward

    def step_reward(self):
        # reward = self.punish_soc()
        reward = self.get_energy_cost() * -1
        return reward

    def step(self, action):
        print(self._quarter_counter)
        self._quarter_counter = self._quarter_counter + 1
        self.step_proceed_quarter()
        # update determined obs was here
        # clip values
        power_from_battery = self.descale_value(action[0], self.range_dict['power_from_battery'][0],
                                                self.range_dict['power_from_battery'][1])
        power_from_battery = np.clip(power_from_battery, self.range_dict["power_from_battery"][0],
                                     self.range_dict["power_from_battery"][1])
        self._power_from_battery = self.scale_value(power_from_battery, self.range_dict['power_from_battery'][0],
                                                    self.range_dict['power_from_battery'][1])
        # action masking
        self.step_soc_battery_new()
        self.step_power_from_grid()
        reward = self.step_reward()
        self.update_determined_obs() # perhaps move this
        self.step_power_from_grid()
        # calculate grid
        self.test_energy_balance()
        info = self._get_info()
        obs = self._get_obs()
        if self._quarter_counter == 96*self.NUMBER_OF_DAYS:
            self.terminated = True
        else:
            self.terminated = False
        return obs, reward, self.terminated, self.truncated, info

    def render(self, mode='human'):
        pass
