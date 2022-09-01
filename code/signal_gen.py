import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
from code import eq_phase_rec
import random
import numpy as np
from math import sqrt
from matplotlib import pyplot

def one_step_integrator(state, ders, inp, noise, dt):
	"""RK4 integrates state with derivative for one step of dt
	
	:param state: state of the variables
	:param ders: derivative functions
	:param inp: scalar input, influence defined through ders
	:param dt: time step
	:return: state after one integration step"""
	D = len(state)
	# 1
	k1 = [ders[i](state, inp) for i in range(D)]
	# 2
	state2 = [state[i]+k1[i]*dt/2.0 for i in range(D)]
	k2 = [ders[i](state2, inp) for i in range(D)]
	# 3
	state3 = [state[i]+k2[i]*dt/2.0 for i in range(D)] 
	k3 = [ders[i](state3, inp) for i in range(D)]
	# 4
	state4 = [state[i]+k3[i]*dt for i in range(D)] 
	k4 = [ders[i](state4, inp) for i in range(D)]
	# put together
	statef = [state[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0*dt + noise*random.gauss(0,1)*sqrt(dt) for i in range(D)]
	return statef

# import forcing
forcing = []
f = open("forcing/forcing.txt", "rt")
for line in f.readlines():
	forcing.append(float(line[0:-1])*parameters.forcing_strength)
f.close()

# choosing timestep
period = eq_phase_rec.oscillator_period(parameters.ders)
dt = period/parameters.signal_timesteps_per_period
N = parameters.signal_length_periods*parameters.signal_timesteps_per_period # total number of timesteps

# generate signal
state = [0.5 for d in range(len(parameters.ders))]
for i in range(int(N/100)): # warmup
	state = one_step_integrator(state, parameters.ders, forcing[i], parameters.noise_strength, dt) 
signal_x = []
signal_y = []
for i in range(N):
	state = one_step_integrator(state, parameters.ders, forcing[i], parameters.noise_strength, dt)
	signal_x.append(state[0])
	signal_y.append(state[1])
#pyplot.plot(forcing)
#pyplot.plot(signal_y)
#pyplot.show()

# write on file
f = open("signal/signal_x.txt", "wt")
for i in range(len(signal_x)):
	f.write(str(signal_x[i])+"\n")
f.close()
f = open("signal/signal_y.txt", "wt")
for i in range(len(signal_y)):
	f.write(str(signal_y[i])+"\n")
f.close()
