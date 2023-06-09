# https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

def scale_eq(value, min_meas, max_meas, min_des,
             max_des):  # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    return ((value - min_meas) / (max_meas - min_meas)) * (max_des - min_des) + min_des
