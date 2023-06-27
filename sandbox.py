# test protocol for environment

# load environment
import gymnasium as gym
import gym_examples
import numpy as np
import math
import matplotlib.pyplot as plt

# test reset
def test_reset_time_obs():
    env = gym.make("gym_examples/EMS_no_gen-v2")
    reset_obs = env.reset()
    day_year = env.descale_value(reset_obs[0]['day_year'], env.range_dict['day_year'][0], env.range_dict['day_year'][1])
    day_week = env.descale_value(reset_obs[0]['day_week'], env.range_dict['day_week'][0], env.range_dict['day_week'][1])
    quarter_day = env.descale_value(reset_obs[0]['quarter_day'], env.range_dict['quarter_day'][0],
                                    env.range_dict['quarter_day'][1])
    # fixed day_year is 99, start quarter is 0
    assert round(day_year[0]) == 99, "should be 99"
    assert round(quarter_day[0]) == 0, "should be 0"
    return

def test_step_time_obs(start_quarter=0, start_day=99):
    env = gym.make("gym_examples/EMS_no_gen-v2")
    # make empty arrays
    day_year_arr = []
    day_week_arr = []
    quarter_day_arr = []

    day_year_check_arr = []
    day_week_check_arr = []
    quarter_day_check_arr = []

    # reset environmenet and get first obs
    env.start_quarter_day = start_quarter
    env.start_day_year = start_day
    reset_obs = env.reset()

    day_year_reset = env.descale_value(reset_obs[0]['day_year'], env.range_dict['day_year'][0], env.range_dict['day_year'][1])
    day_week_reset = env.descale_value(reset_obs[0]['day_week'], env.range_dict['day_week'][0], env.range_dict['day_week'][1])
    quarter_day_reset = env.descale_value(reset_obs[0]['quarter_day'], env.range_dict['quarter_day'][0],
                                    env.range_dict['quarter_day'][1])
    # append first to array
    day_year_arr = np.append(day_year_arr, day_year_reset)
    day_week_arr = np.append(day_week_arr, day_week_reset)
    quarter_day_arr = np.append(quarter_day_arr, quarter_day_reset)

    day_year_check = day_year_reset
    day_week_check = day_week_reset
    quarter_day_check = quarter_day_reset

    day_year_check_arr = np.append(day_year_check_arr, day_year_check)
    day_week_check_arr = np.append(day_week_check_arr, day_week_check)
    quarter_day_check_arr = np.append(quarter_day_check_arr, quarter_day_check)

    # loop 14 days
    counter = 1
    while (env.terminated == False):
        step_obs = env.step([0])
        day_year = np.round(env.descale_value(step_obs[0]['day_year'], env.range_dict['day_year'][0],
                                     env.range_dict['day_year'][1]))
        day_week = np.round(env.descale_value(step_obs[0]['day_week'], env.range_dict['day_week'][0],
                                     env.range_dict['day_week'][1]))
        quarter_day = np.round(env.descale_value(step_obs[0]['quarter_day'], env.range_dict['quarter_day'][0],
                                        env.range_dict['quarter_day'][1]))

        day_year_arr = np.append(day_year_arr, day_year)
        day_week_arr = np.append(day_week_arr, day_week)
        quarter_day_arr = np.append(quarter_day_arr, quarter_day)

        # check values
        curr_day_year = np.round(math.floor(counter/96) + day_year_reset)
        day_year_check_arr = np.append(day_year_check_arr, curr_day_year)
        day_week_check_arr = np.append(day_week_check_arr, curr_day_year % 7)
        quarter_day_check_arr = np.append(quarter_day_check_arr, (counter+quarter_day_reset)%96)

        counter = counter+1


    assert np.array_equal(day_year_arr, day_year_check_arr), "day year must be equal"
    assert np.array_equal(day_week_arr, day_week_check_arr), "day week must be equal"
    assert np.array_equal(quarter_day_arr, quarter_day_check_arr), "quarter day must be equal"
    return

def test_reset_SOC():
    initial_socs = np.arange(-1, 1.25, 0.25)
    scaled_soc_arr = []
    scaled_soc_check_arr = []
    unscaled_soc_arr = []
    unscaled_soc_check_arr = []
    for initial_soc in initial_socs:
        env = gym.make("gym_examples/EMS_no_gen-v2")
        env.test_set_initial_soc(initial_soc)
        # scaled
        scaled_soc_arr = np.append(scaled_soc_arr, env.reset()[0]['soc_battery'])
        scaled_soc_check_arr = np.append(np.clip(scaled_soc_check_arr, -1, 1), initial_soc)
        # unscaled
        unscaled_soc_arr = np.append(unscaled_soc_arr, env.get_prev_scaled_obs()['soc_battery'])
        unscaled_soc_check_arr = np.append(unscaled_soc_check_arr, env.descale_value(initial_soc, env.range_dict['soc_battery'][0], env.range_dict['soc_battery'][1]))

    assert np.array_equal(scaled_soc_arr, scaled_soc_check_arr), "scaled SOC must be equal"
    assert np.array_equal(unscaled_soc_arr, unscaled_soc_check_arr), "unscaled SOC must be equal"
    return


# test set power from battery
# make env
env = gym.make("gym_examples/EMS_no_gen-v2")
env.reset()

soc_arr = []
soc_check_arr = []
set_power_from_battery_original_arr = []
set_power_from_battery_check_arr = []
set_power_from_battery_arr = []

counter = 0
while (env.terminated == False):
    # determine previous state
    prev_soc = env.get_prev_scaled_obs()['soc_battery']
    # determine power from battery
    #set_power_from_battery = np.random.uniform(env.range_power_from_battery[0], env.range_power_from_battery[1])
    #set_power_from_battery_scaled = env.scale_value(set_power_from_battery, env.range_power_from_battery[0], env.range_power_from_battery[1])
    f = 1/500
    set_power_from_battery_scaled = 0.1*math.cos(counter*f*2*math.pi)
    set_power_from_battery_descaled = env.descale_value(set_power_from_battery_scaled, env.range_power_from_battery[0], env.range_power_from_battery[1])
    # step the environment
    env.step([set_power_from_battery_scaled])
    # determine observed and reference soc
    curr_soc = env.get_prev_scaled_obs()['soc_battery']
    curr_soc_check_unclipped = prev_soc - 0.25*set_power_from_battery_descaled
    curr_soc_check = np.clip(curr_soc_check_unclipped, 0, env.battery_capacity)
    # determine allowed power from battery
    curr_power_from_battery = env.power_from_battery
    curr_power_from_battery_check = set_power_from_battery_descaled
    if curr_soc_check_unclipped < 0:
        delta_power_from_battery = (0 - curr_soc_check_unclipped)*4
        curr_power_from_battery_check = curr_power_from_battery_check - delta_power_from_battery
    if curr_soc_check_unclipped > 4e5:
        delta_power_from_battery = (4e5-curr_soc_check_unclipped)*4
        curr_power_from_battery_check = curr_power_from_battery_check - delta_power_from_battery
    curr_power_from_battery_check = env.scale_value(curr_power_from_battery_check, env.range_power_from_battery[0], env.range_power_from_battery[1])
    # extend arrays
    soc_arr = np.append(soc_arr, round(curr_soc,3))
    soc_check_arr = np.append(soc_check_arr, round(curr_soc_check,3))
    set_power_from_battery_original_arr = np.append(set_power_from_battery_original_arr, set_power_from_battery_scaled)
    set_power_from_battery_arr = np.append(set_power_from_battery_arr, round(curr_power_from_battery,5))
    set_power_from_battery_check_arr = np.append(set_power_from_battery_check_arr, round(curr_power_from_battery_check,5))
    counter=counter+1

assert np.array_equal(soc_arr, soc_check_arr), "calculated SOC must be equal"

