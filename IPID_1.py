import numpy as np
import numba as nb

@nb.njit
def f_from_modes(phi, Fourier_modes):
	'''computes function from Fourier modes'''
	N_Fourier = int((len(Fourier_modes)+1)/2)
	
	res = Fourier_modes[0]
	for n in range(1, N_Fourier):
		res += Fourier_modes[2*n-1]*np.cos(n*phi)
		res += Fourier_modes[2*n  ]*np.sin(n*phi)
	return res

def f_from_modes_array(Fourier_modes, samples = 200):
	#args = np.linspace(0, 2*np.pi, samples)
	return [f_from_modes(arg, Fourier_modes) for arg in np.linspace(0, 2*np.pi, samples)]

def thresholding_signal(signal, threshold, direction=1):
	'''
	thresholds a signal at "threshold"
	returns events
	'''
	
	events = []
	# checking if threshold is within signals range

	# construct 
	check_signal = direction*(signal-threshold)

	# thresholding
	for i in range(5,len(signal)-6):
		# four points, 5 below and 5 above threshold to count
		sdif1 = check_signal[i-5]
		sdif2 = check_signal[i]
		sdif3 = check_signal[i+1]
		sdif4 = check_signal[i+6]
		if(sdif2*sdif3 < 0 and sdif1*sdif4 < 0):
			if(sdif3 > 0 and sdif4 > 0):
				#events.append(i+abs(sdif2)/(abs(sdif2)+abs(sdif3)))
				events.append(i)
	return np.array(events)

###################
### phase / PRC ###
###################

def initialize_phase(events, length):
	# fill with "-1"
	phase = -np.ones(length)

	# between events: linear
	for i in range(len(events)-1):
		T = events[i+1]-events[i]
		for t in range(events[i], events[i+1]):
			phase[t] = 2.*np.pi*(t-events[i])/T
	return phase

@nb.njit(parallel=True)
def A_PRC_prep(events, phase, forcing, dt, N_Fourier):
	A = np.zeros((len(events)-1, 2*N_Fourier)) 

	for i in nb.prange(len(events)-1):
		# period/natural frequency
		A[i,0] = (events[i+1]-events[i])*dt
		
		# Z_0
		A[i,1] = np.trapz(forcing[events[i]:events[i+1]], dx=dt)
		
		for n in nb.prange(1,N_Fourier):
			# cos
			A[i,2*n]   = np.trapz(np.cos(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			# sin
			A[i,2*n+1] = np.trapz(np.sin(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
	return  A 

def error_Z0(events):
	'''
	calculates error E_Z0
	'''
	Ts = [events[i+1]-events[i] for i in range(len(events)-1)]
	avg_T = np.average(Ts)
	avg_w = 2*np.pi/avg_T
	diffs = [(avg_w*T-2*np.pi)**2 for T in Ts]
	return np.sqrt(np.average(diffs))

def PRC_infer_step(events, phase, forcing, dt, N_Fourier):
	'''one iteration of PRC inference 
	calculates the Fourier moments of the PRC contained in "sol" by
	b = A*sol

	sol[0]: 	natural freq. omega
	sol[1]:		Z_0
	sol[2*n]: 	Z_n^cos
	sol[2*n+1]: Z_n^sin
	'''

	# prepare matrix "A" with the fourier integrals
	A = A_PRC_prep(events, phase, forcing, dt, N_Fourier)

	# prepare vector "b"
	b = np.array([2*np.pi for i in range(len(events)-1)]).transpose()

	# minimization
	sol = np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b))

	omega = sol[0]
	PRC_modes = np.array(sol[1:])

	# phase recalculation
	Phi_n = []
	new_phase = -np.ones(len(forcing))

	for i in range(len(events)-1):
		# Euler scheme
		new_phase[events[i]] = 0.
		for t in range(events[i] + 1, events[i+1] + 1):
			new_phase[t] = new_phase[t-1] + (omega + f_from_modes(new_phase[t-1], PRC_modes)*forcing[t-1])*dt
			
		# phase at the end
		phi_end = new_phase[events[i+1]]

		# store for error calculation
		Phi_n.append(phi_end)
		
		# rescale so its 2pi at the end
		for t in range(events[i], events[i+1]):
			new_phase[t] = 2*np.pi*new_phase[t]/phi_end

	# error calculation
	diffs = [(psi-2*np.pi)**2 for psi in Phi_n]
	error = np.sqrt(np.average(diffs))
	
	return omega, PRC_modes, new_phase, error

#################################
### isostable amplitude / IRC ###
#################################

def initialize_psi(signal_at_events, events, length):
	# fill with "0"
	psi = np.zeros(length)
	
	# average
	x0 = np.average(signal_at_events)

	# between events: linear
	for i in range(len(events)-1):
		a0 = signal_at_events[i]
		a1 = signal_at_events[i+1]
		T = events[i+1]-events[i]
		for t in range(events[i], events[i+1]):
			psi[t] = a0+(t-events[i])*(a1-a0)/T - x0
	return psi

@nb.njit(parallel=True)
def A_IRC_prep(events, phase, psi, forcing, x0, dt, N_Fourier):
	A = np.zeros((len(events)-1, 2*N_Fourier+1))

	for i in nb.prange(len(events)-1):
		# kappa
		A[i,0] = np.trapz(psi[events[i]:events[i+1]] + x0, dx=dt)

		# Z_0
		A[i,1] = np.trapz(forcing[events[i]:events[i+1]], dx=dt)

		for n in nb.prange(1,N_Fourier):
			# cos
			A[i,2*n]   = np.trapz(np.cos(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)
			# sin
			A[i,2*n+1] = np.trapz(np.sin(n*phase[events[i]:events[i+1]])*forcing[events[i]:events[i+1]], dx=dt)

		# kappa*x0
		A[i,2*N_Fourier] = -(events[i+1]-events[i])*dt
	return A

def error_I0(signal_at_events):
	'''
	calculates error E_I0
	'''

	#a_sqs = [(s_n-x0)**2 for s_n in signal_at_events]
	#return np.sqrt(np.average(a_sqs))
	return np.std(signal_at_events)

def IRC_infer_step(events, phase, psi, signal_at_events, forcing, x0, dt, N_Fourier):
	'''one iteration of IRC inference 
	calculates the Fourier moments of the IRC contained in "sol" by
	b = A*sol

	sol[0]: 	kappa
	sol[1]:		I_0
	sol[2*n]: 	I_n^cos
	sol[2*n+1]: I_n^sin
	sol[-1]: 	kappa*x0
	'''

	# prepare matrix "A" with the fourier integrals
	A = A_IRC_prep(events, phase, psi, forcing, x0, dt, N_Fourier)

	# prepare vector "b"
	b = np.array([signal_at_events[i+1]-signal_at_events[i] for i in range(len(events)-1)]).transpose()

	# minimization
	sol = np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b))

	# unpack solution
	kappa = sol[0]
	x0_new = sol[-1]/kappa # k*x0/k
	IRC_modes = np.array(sol[1:-1])

	# isostable recalculation
	Psi_n = []
	new_psi = np.zeros(len(forcing))

	for i in range(len(events)-1):
		# Euler scheme (can alternatively be calculated by an integral)
		new_psi[events[i]] = signal_at_events[i]-x0_new
		for t in range(events[i] + 1, events[i+1] + 1):
			new_psi[t] = new_psi[t-1] + (kappa*new_psi[t-1] + f_from_modes(phase[t],IRC_modes)*forcing[t])*dt

		# isostable amplitude at the end
		psi_end = new_psi[events[i+1]]

		# store for error calculation
		Psi_n.append(psi_end)

	# error calculation
	diffs = [(Psi_n[i]-(signal_at_events[i+1]-x0_new))**2 for i in range(len(Psi_n))]
	error =np.sqrt(np.average(diffs))

	return kappa, x0_new, IRC_modes, new_psi, error

###################################
### inference of phase dynamics ###
###################################

def infer_phase_respones(signal, forcing, dt, threshold_signal=0., direction=1, iterations=8, N_Fourier=10, SpeakToMe=True):

	signal = np.array(signal)
	forcing = np.array(forcing)

	### phase ###

	if SpeakToMe==True: print('thresholding signal')
	events = thresholding_signal(signal, threshold_signal, direction=direction)

	E_Z0 = error_Z0(events)

	if SpeakToMe==True: print('initialize phase')
	phase = [initialize_phase(events, len(signal))]

	if SpeakToMe==True: print('iterating PRC inference')
	omega = []
	PRC_modes = []
	E_Z = []

	for i in range(iterations):
		omega_new, PRC_modes_new, phase_new, E_Z_new  = PRC_infer_step(events, phase[-1], forcing, dt, N_Fourier)
		
		if SpeakToMe==True: print('i='+ str(i) + ': E_Z/E_Z0='+ str(np.round(E_Z_new/E_Z0, 5)))
		
		omega.append(omega_new)
		PRC_modes.append(PRC_modes_new)
		phase.append(phase_new)
		E_Z.append(E_Z_new)

	return omega, PRC_modes, phase, E_Z, E_Z0

def infer_isostable_respones(signal, forcing, dt, phase, threshold_phase=np.pi, direction=1, iterations=8, N_Fourier=10, SpeakToMe=True):
	
	signal = np.array(signal)
	forcing = np.array(forcing)

	if SpeakToMe==True: print('thresholding phase')
	events = thresholding_signal(phase, threshold_phase, direction=direction)
	
	signal_at_events = signal[events]
	E_I0 = error_I0(signal_at_events)
	
	if SpeakToMe==True: print('initialize isostable amplitude')
	psi = [initialize_psi(signal_at_events, events, len(signal))]
	
	if SpeakToMe==True: print('iterating IRC inference')

	kappa = []
	IRC_modes = []
	x0 = [np.average(signal_at_events)]
	E_I  = []

	for i in range(iterations):
		kappa_new, x0_new, IRC_modes_new, psi_new, E_I_new  = IRC_infer_step(events, phase, psi[-1], signal_at_events, forcing, x0[-1], dt, N_Fourier)

		if SpeakToMe==True: print('i='+ str(i) + ': E_I/E_I0='+ str(np.round(E_I_new/E_I0, 5)))
		
		kappa.append(kappa_new)
		IRC_modes.append(IRC_modes_new)
		x0.append(x0_new)
		psi.append(psi_new)
		E_I.append(E_I_new)

	return kappa, IRC_modes, x0, psi, E_I, E_I0

#########################
### quick "do it all" ###
#########################

def infer_response_curves(*args, params_phase = {}, params_iso = {}):

	### phase ###

	omega, PRC_modes, phase, E_Z, E_Z0 = infer_phase_respones(*args, **params_phase)

	### isostable amplitude ###

	kappa, IRC_modes, x0, psi, E_I, E_I0 = infer_isostable_respones(*args, phase[-1], **params_iso)

	return omega, PRC_modes, phase, kappa, IRC_modes, x0, psi