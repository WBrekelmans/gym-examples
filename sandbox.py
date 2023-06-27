# test protocol for environment

# load environment
import gymnasium as gym
import gym_examples
import numpy as np
import math

env = gym.make("gym_examples/EMS_no_gen-v2")


# test reset
def test_reset_time_obs():
    reset_obs = env.reset()
    day_year = env.descale_value(reset_obs[0]['day_year'], env.range_dict['day_year'][0], env.range_dict['day_year'][1])
    day_week = env.descale_value(reset_obs[0]['day_week'], env.range_dict['day_week'][0], env.range_dict['day_week'][1])
    quarter_day = env.descale_value(reset_obs[0]['quarter_day'], env.range_dict['quarter_day'][0],
                                    env.range_dict['quarter_day'][1])
    # fixed day_year is 99, start quarter is 0
    assert round(day_year[0]) == 99, "should be 99"
    assert round(quarter_day[0]) == 0, "should be 0"
    return

def test_step_time_obs():
    # make empty arrays
    day_year_arr = []
    day_week_arr = []
    quarter_day_arr = []

    day_year_check_arr = []
    day_week_check_arr = []
    quarter_day_check_arr = []

    # reset environmenet and get first obs
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
    for i in range(14*24*4):
        step_obs = env.step([0])
        day_year = env.descale_value(step_obs[0]['day_year'], env.range_dict['day_year'][0],
                                     env.range_dict['day_year'][1])
        day_week = env.descale_value(step_obs[0]['day_week'], env.range_dict['day_week'][0],
                                     env.range_dict['day_week'][1])
        quarter_day = env.descale_value(step_obs[0]['quarter_day'], env.range_dict['quarter_day'][0],
                                        env.range_dict['quarter_day'][1])

        day_year_arr = np.append(day_year_arr, day_year)
        day_week_arr = np.append(day_week_arr, day_week)
        quarter_day_arr = np.append(quarter_day_arr, quarter_day)

        # check values
        curr_day_year = np.round(math.floor(counter/96) + day_year_reset)
        day_year_check_arr = np.append(day_year_check_arr, curr_day_year)
        day_week_check_arr = np.append(day_week_check_arr, curr_day_year % 7)
        quarter_day_check_arr = np.append(quarter_day_check_arr, (counter+quarter_day_reset)%96)

        counter = counter+1
        assert day_year_check_arr == day_year_arr, "must be equal"
        assert round(day_year[0]) == 99, "should be 99"
        assert round(day_year[0]) == 99, "should be 99"


for i in range(0,10):
    print(i%7)


reset_obs = env.reset()
