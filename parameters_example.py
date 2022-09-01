from math import sin, cos, sqrt, atan, pi

# system
period = 25.17
def dx(state, inp=0):
	return 0.25*state[1] - 0.075*sin(state[1])*state[0]/2 # sincos
	return state[1] - state[0]/2 + inp # cauchy
def dy(state, inp=0):
	return -0.25*state[0] + 0.075*cos(state[0])*state[1]/2 + inp # sincos
	return -state[0] + state[1]/(1+state[0]**2) # cauchy
ders = [dx,dy]

# angle function
def angl(x,y):
	if(x == 0):
		if(y > 0):
			return pi/2
		return 3*pi/2
	elif(x < 0):
		return atan(y/x)+pi
	elif(y < 0):
		return atan(y/x)+2*pi
	return atan(y/x)

# signal generate
signal_timesteps_per_period = int(2*pi*100) 
signal_length_periods = 1000
noise_strength = 0.001*0

# forcing
pulses_per_period = 8 # roughly
forcing_pulses = int(1.5*signal_length_periods*pulses_per_period) # 1.5* so signal_gen does not run out of forcing
pulse_up_steps = 15
pulse_pause_steps = 30
pulse_down_steps = 80
forcing_strength = 15

# thresholding
threshold = 0
phase_threshold = pi/2-0.3

# prc inference
prc_N_fourier = 8

# plot
prc_scale = signal_timesteps_per_period/period/forcing_strength 
amp_c = 0.5 # amplitude constant
arc_scale = amp_c*prc_scale
phase_shift = 0
