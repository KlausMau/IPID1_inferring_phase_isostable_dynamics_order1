from math import pi, sin, cos, ceil, floor, sqrt
import numpy as np
import numba as nb

# compute function from Fourier modes

def f_from_sol(fi, sol):
		#N_Fourier = int((len(sol)-1)/2)
		#res = float(sol[1])
		N_Fourier = int(len(sol)/2)
		res = 0.
		#for n in range(1, N_Fourier):
		for n in range(1, N_Fourier+1):
			res += float(sol[2*n   -2])*cos(n*fi)
			res += float(sol[2*n+1 -2])*sin(n*fi)
		return res

def f_from_sol_array(sol, samples = 200):
	#args = [resolution*i for i in range(int(2*pi/resolution))] 
	args = np.linspace(0, 2*np.pi, samples)
	#return f_from_sol(args, sol)
	return [f_from_sol(arg,sol) for arg in args]

# inference steps

def PRC_infer_step(events, phase, forcing, N_Fourier):

	# calculates the Fourier moments of the PRC ("sol")

	A = [[] for i in range(len(events)-1)]
	b = np.matrix([2*pi for i in range(len(events)-1)]).transpose()
	for i in range(len(events)-1):
		A[i].append(events[i+1]-events[i]) # period
		# fourier integrals
		integral = forcing[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
		for t in range(ceil(events[i]),floor(events[i+1])):
			integral += forcing[t]
		integral += forcing[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
		A[i].append(integral) # natural frequency
		for n in range(1,N_Fourier):
			# cos
			integral = forcing[ceil(events[i])-1]*cos(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep

			for t in range(ceil(events[i]),floor(events[i+1])):	
				integral += forcing[t]*cos(n*phase[t])
			#print("index = "+str(floor(events[i+1]))+" and len(phase) = "+str(len(phase)))
			integral += forcing[floor(events[i+1])]*cos(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			A[i].append(integral)
			
			# sin
			integral = forcing[ceil(events[i])-1]*sin(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
			for t in range(ceil(events[i]),floor(events[i+1])):
				integral += forcing[t]*sin(n*phase[t])
			integral += forcing[floor(events[i+1])]*sin(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			A[i].append(integral)

	# minimization
	A = np.matrix(A)
	AT = A.transpose()
	ATA = AT*A
	ATAinv = np.linalg.inv(ATA)
	sol = ATAinv*(AT*b)

	# phase recalculation
	new_phase = []
	psis = []
	for t in range(floor(events[0])+1):
		new_phase.append(-1)
	for i in range(len(events)-1):
		ph = (sol[0]+f_from_sol(0,sol)*forcing[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
		new_phase.append(ph)
		for t in range(ceil(events[i]),floor(events[i+1])):
			ph = ph + sol[0] + f_from_sol(ph,sol)*forcing[t]
			new_phase.append(ph)
		# phase at the end
		psi = ph + (sol[0]+f_from_sol(ph,sol)*forcing[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
		psis.append(psi)
		# rescale so its 2pi at the end
		for t in range(ceil(events[i]),floor(events[i+1])+1):
			new_phase[t] = 2*pi*new_phase[t]/psi

	# error
	diffs = [(psi-2*pi)**2 for psi in psis]
	error = sqrt(np.average(diffs))
	# 0 error
	Ts = [events[i+1]-events[i] for i in range(len(events)-1)]
	avg_T = np.average(Ts)
	avg_w = 2*pi/avg_T
	diffs = [(avg_w*T-2*pi)**2 for T in Ts]
	error0 = sqrt(np.average(diffs))

	return sol, new_phase, error, error0

# thresholding

def thresholding_signal(signal, threshold):
	events = []
	# checking if threshold is within signals range

	# thresholding
	for i in range(5,len(signal)-6):
		# four points, 5 below and 5 above threshold to count
		sdif1 = signal[i-5]-threshold
		sdif2 = signal[i]-threshold
		sdif3 = signal[i+1]-threshold
		sdif4 = signal[i+6]-threshold
		if(sdif2*sdif3 < 0 and sdif1*sdif4 < 0):
			if(sdif3 > 0 and sdif4 > 0):
				events.append(i+abs(sdif2)/(abs(sdif2)+abs(sdif3)))
	return events

# initialization

def initialize_phase(events):
	
	phase = []

	for t in range(floor(events[0])):
		phase.append(-1.)

	for i in range(len(events)-1):
		T = events[i+1]-events[i]
		for t in range(ceil(events[i]), floor(events[i+1])+1):
			phase.append(2*pi*(t-events[i])/T)
	
	phase.append(-1.)

	return phase

# main inferrence routine

def infer_response_curves(signal, forcing, threshold=0., iterations = 8, N_Fourier=10):

	print('thresholding signal')
	events = thresholding_signal(signal, threshold)
	#print(events)

	print('initialize phase')
	phase = []
	phase.append(initialize_phase(events))
	#print(phase)

	print('iterating PRC inference')

	#f  = nb.njit(PRC_infer_step)

	sol = []
	for i in range(iterations):
		print('i='+ str(i))
		
		sol_new, phase_new, error, error0  = PRC_infer_step(events, phase[-1], forcing, N_Fourier)
		#sol_new, phase_new, error, error0  = f(events, phase[-1], forcing, N_Fourier)
		
		print('E/E0='+ str(np.round(error/error0, 5)))
		
		sol.append(sol_new)
		phase.append(phase_new)

	return sol, phase