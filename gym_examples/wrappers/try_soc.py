capacity = 4e5 # Watt hours
charge_rate = 4e5 # Watt hours

power_from_battery = 4e5
soc = capacity
soc_new = capacity - 0.25*power_from_battery
env.reset()