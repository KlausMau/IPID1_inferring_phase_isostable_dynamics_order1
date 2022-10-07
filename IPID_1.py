import numpy as np
import numba as nb

# for their integer output
#from math import floor, ceil

# compute function from Fourier modes

#@nb.njit
def f_from_sol(fi, sol):
	N_Fourier = int((len(sol)-1)/2)
	res = float(sol[1])
	#N_Fourier = int(len(sol)/2)
	#res = 0.
	
	for n in range(1, N_Fourier):
	#for n in range(1, N_Fourier+1):
		res += float(sol[2*n  ])*np.cos(n*fi)
		res += float(sol[2*n+1])*np.sin(n*fi)
	return res

def f_from_sol_array(sol, samples = 200):
	#args = [resolution*i for i in range(int(2*pi/resolution))] 
	args = np.linspace(0, 2*np.pi, samples)
	#return f_from_sol(args, sol)
	return [f_from_sol(arg,sol) for arg in args]

# inference steps

def PRC_infer_step(events, phase, forcing, dt, N_Fourier):

	# calculates the Fourier moments of the PRC ("sol") contained in "z" by
	# b = A*z

	# sol[0]: 		period/natural freq.
	# sol[1]:		Z_0
	# sol[2*n]: 	Z_n^cos
	# sol[2*n+1]: 	Z_n^sin

	# prepare matrix "A" with the fourier integrals
		
	A = [[] for i in range(len(events)-1)]
	
	for i in range(len(events)-1):
		
		A[i].append((events[i+1]-events[i])*dt) # period/natural frequency

		#integral = forcing[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
		#for t in range(ceil(events[i]), floor(events[i+1])):
		#	integral += forcing[t]
		#integral += forcing[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
		integral = np.trapz(forcing[events[i]:events[i+1]], dx=dt)
		
		A[i].append(integral) # Z_0
		
		for n in range(1,N_Fourier):
			# cos
			#integral = forcing[ceil(events[i])-1]*np.cos(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep

			#for t in range(ceil(events[i]),floor(events[i+1])):	
			#	integral += forcing[t]*np.cos(n*phase[t])
			#print("index = "+str(floor(events[i+1]))+" and len(phase) = "+str(len(phase)))
			#integral += forcing[floor(events[i+1])]*np.cos(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			
			integral = np.trapz(np.cos(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			A[i].append(integral)
			
			# sin
			#integral = forcing[ceil(events[i])-1]*np.sin(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
			#for t in range(ceil(events[i]), floor(events[i+1])):
		#		integral += forcing[t]*np.sin(n*phase[t])
			#integral += forcing[floor(events[i+1])]*np.sin(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
			
			integral = np.trapz(np.sin(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			A[i].append(integral)

	A_np = np.matrix(A, dtype=float)


	# prepare vector "b"
	b = np.matrix([2*np.pi for i in range(len(events)-1)]).transpose()

	# minimization
	sol = np.linalg.solve(A_np.transpose().dot(A_np), A_np.transpose().dot(b))


	#A = np.matrix(A, dtype=float)
	#AT = A.transpose()
	#ATA = AT*A

	#print(np.shape(ATA))
#	print(ATA.shape())

	#ATAinv = np.linalg.inv(ATA)
	#sol = ATAinv*(AT*b)

	#sol = np.linalg.inv(ATA)*(AT*b)

	# phase recalculation
	psis = []
	
	#new_phase = []
	#for t in range(floor(events[0])+1):
	#	new_phase.append(-1)
	new_phase = -np.ones(len(forcing))


	for i in range(len(events)-1):
		#ph = (sol[0]+f_from_sol(0,sol)*forcing[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
		#new_phase.append(ph)
		
		#for t in range(ceil(events[i]),floor(events[i+1])):
		new_phase[events[i]] = 0.
		for t in range(events[i] + 1, events[i+1] + 1):
			#ph = ph + sol[0] + f_from_sol(ph,sol)*forcing[t]
			new_phase[t] = new_phase[t-1] + (sol[0] + f_from_sol(new_phase[t-1],sol)*forcing[t-1])*dt
			#new_phase.append(ph)

		# phase at the end
		#psi = ph + (sol[0]+f_from_sol(ph,sol)*forcing[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
		psi = new_phase[events[i+1]]
		psis.append(psi)
		
		# rescale so its 2pi at the end
		#for t in range(ceil(events[i]), floor(events[i+1])+1):
		for t in range(events[i], events[i+1]):
			new_phase[t] = 2*np.pi*new_phase[t]/psi

	# error calculation
	diffs = [(psi-2*np.pi)**2 for psi in psis]
	error = np.sqrt(np.average(diffs))
	
	return sol, new_phase, error

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
	return events

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

	print('thresholding signal')
	events = thresholding_signal(signal, threshold)

	E0 = error_0(events) 

	print('initialize phase')
	phase = []
	phase.append(initialize_phase(events, len(signal)))
	#print(phase)

	print('iterating PRC inference')

	#f  = nb.njit(PRC_infer_step)

	sol = []
	for i in range(iterations):
		print('i='+ str(i))
		sol_new, phase_new, error  = PRC_infer_step(events, phase[-1], forcing, dt, N_Fourier)
		#sol_new, phase_new, error  = f(events, phase[-1], forcing, N_Fourier)
		
		print('E/E0='+ str(np.round(error/E0, 5)))
		
		sol.append(sol_new)
		phase.append(phase_new)

	return sol, phase