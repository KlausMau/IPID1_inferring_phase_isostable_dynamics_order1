import numpy as np
import numba as nb

# compute function from Fourier modes

@nb.njit
def f_from_modes(phi, Fourier_modes):
	N_Fourier = int((len(Fourier_modes)+1)/2)
	
	res = Fourier_modes[0]
	for n in range(1, N_Fourier):
		res += Fourier_modes[2*n-1]*np.cos(n*phi)
		res += Fourier_modes[2*n  ]*np.sin(n*phi)
	return res

def f_from_modes_array(Fourier_modes, samples = 200):
	#args = np.linspace(0, 2*np.pi, samples)
	return [f_from_modes(arg, Fourier_modes) for arg in np.linspace(0, 2*np.pi, samples)]

# Numba accelerated functions

@nb.njit(parallel=True)
def A_prep(events, phase, forcing, dt, N_Fourier):
	#A = [[] for i in range(len(events)-1)]
	
	A = np.zeros((len(events)-1, 2*N_Fourier)) #, dtype=float)

	for i in nb.prange(len(events)-1):
		
		#A[i].append((events[i+1]-events[i])*dt) # period/natural frequency
		A[i,0] = (events[i+1]-events[i])*dt
		#integral = forcing[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
		#for t in range(ceil(events[i]), floor(events[i+1])):
		#	integral += forcing[t]
		#integral += forcing[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
		#integral = np.trapz(forcing[events[i]:events[i+1]], dx=dt)
		#A[i].append(integral) # Z_0

		A[i,1] = np.trapz(forcing[events[i]:events[i+1]], dx=dt)
		
		for n in nb.prange(1,N_Fourier):
			# cos
			#integral = forcing[ceil(events[i])-1]*np.cos(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep

			#for t in range(ceil(events[i]),floor(events[i+1])):	
			#	integral += forcing[t]*np.cos(n*phase[t])
			#print("index = "+str(floor(events[i+1]))+" and len(phase) = "+str(len(phase)))
			#integral += forcing[floor(events[i+1])]*np.cos(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			
			#integral = np.trapz(np.cos(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			#A[i].append(integral)
			A[i,2*n] = np.trapz(np.cos(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			
			# sin
			#integral = forcing[ceil(events[i])-1]*np.sin(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
			#for t in range(ceil(events[i]), floor(events[i+1])):
			#		integral += forcing[t]*np.sin(n*phase[t])
			#integral += forcing[floor(events[i+1])]*np.sin(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			
			#integral = np.trapz(np.sin(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			A[i,2*n+1] = np.trapz(np.sin(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			
			#A[i].append(integral)

	return  A 

# inference steps

def PRC_infer_step(events, phase, forcing, dt, N_Fourier):

	# calculates the Fourier moments of the PRC contained in "sol" by
	# b = A*sol

	# sol[0]: 		period/natural freq.
	# sol[1]:		Z_0
	# sol[2*n]: 	Z_n^cos
	# sol[2*n+1]: 	Z_n^sin

	# prepare matrix "A" with the fourier integrals
	A = A_prep(events, phase, forcing, dt, N_Fourier)

	# prepare vector "b"
	b = np.array([2*np.pi for i in range(len(events)-1)]).transpose()

	# minimization
	sol = np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b))

	omega = sol[0]
	PRC_modes = np.array(sol[1:])

	# phase recalculation
	psis = []
	new_phase = -np.ones(len(forcing))

	for i in range(len(events)-1):
		# Euler scheme
		new_phase[events[i]] = 0.
		for t in range(events[i] + 1, events[i+1] + 1):
			new_phase[t] = new_phase[t-1] + (omega + f_from_modes(new_phase[t-1], PRC_modes)*forcing[t-1])*dt
			
		# phase at the end
		psi = new_phase[events[i+1]]
		psis.append(psi)
		
		# rescale so its 2pi at the end
		for t in range(events[i], events[i+1]):
			new_phase[t] = 2*np.pi*new_phase[t]/psi

	# error calculation
	diffs = [(psi-2*np.pi)**2 for psi in psis]
	error = np.sqrt(np.average(diffs))
	
	return omega, PRC_modes, new_phase, error

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
				#events.append(i+abs(sdif2)/(abs(sdif2)+abs(sdif3)))
				events.append(i)
	return np.array(events)

def error_0(events):
	Ts = [events[i+1]-events[i] for i in range(len(events)-1)]
	avg_T = np.average(Ts)
	avg_w = 2*np.pi/avg_T
	diffs = [(avg_w*T-2*np.pi)**2 for T in Ts]
	return np.sqrt(np.average(diffs))

# initialization

def initialize_phase(events, length):
	
	#phase = []

	#for t in range(floor(events[0])):
	#	phase.append(-1.)

	# fill with "-1"
	phase = -np.ones(length)

	# in between events: linear
	for i in range(len(events)-1):
		T = events[i+1]-events[i]
		#for t in range(ceil(events[i]), floor(events[i+1])+1):
		for t in range(events[i], events[i+1]):
			#phase.append(2*np.pi*(t-events[i])/T)
			phase[t] = 2.*np.pi*(t-events[i])/T

	#phase.append(-1.)

	return phase

# main inferrence routine

def infer_response_curves(signal, forcing, dt, threshold=0., iterations = 8, N_Fourier=10):

	forcing = np.array(forcing)

	print('thresholding signal')
	events = thresholding_signal(signal, threshold)

	E0 = error_0(events) 

	print('initialize phase')
	phase = []
	phase.append(initialize_phase(events, len(signal)))

	print('iterating PRC inference')

	omega = []
	PRC_modes = []

	for i in range(iterations):
		print('i='+ str(i))
		omega_new, PRC_modes_new, phase_new, error  = PRC_infer_step(events, phase[-1], forcing, dt, N_Fourier)
		
		print('E/E0='+ str(np.round(error/E0, 5)))
		
		omega.append(omega_new)
		PRC_modes.append(PRC_modes_new)
		phase.append(phase_new)

	return omega, PRC_modes, phase