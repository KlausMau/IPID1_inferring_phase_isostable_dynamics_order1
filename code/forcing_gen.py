import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
import random
import numpy as np
from matplotlib import pyplot

# random times between pulses
ran_num = [int(parameters.signal_timesteps_per_period/parameters.pulses_per_period*np.random.poisson(lam=3.0)/3.0) for i in range(parameters.forcing_pulses)] # Poissonian
ran_num = [int(parameters.signal_timesteps_per_period/parameters.pulses_per_period*(1+np.random.poisson(lam=3.0)/3.0)) for i in range(parameters.forcing_pulses)] # rare Poissonian
#pyplot.plot(np.histogram(ran_num)[0])
#pyplot.show()

# generate
forcing = []
for i in range(parameters.forcing_pulses):
	# time between pulses
	for r in range(ran_num[i]):
		forcing.append(0)
	# pulse
	for r in range(parameters.pulse_up_steps):
		forcing.append(1/parameters.pulse_up_steps)
	for r in range(parameters.pulse_pause_steps):
		forcing.append(0)
	for r in range(parameters.pulse_down_steps):
		forcing.append(-1/parameters.pulse_down_steps)
#pyplot.plot(forcing)
#pyplot.show()

# write on file
f = open("forcing/forcing.txt", "wt")
for i in range(len(forcing)):
	f.write(str(forcing[i])+"\n")
f.close()
