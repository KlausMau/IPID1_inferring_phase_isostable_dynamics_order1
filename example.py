import numpy as np
import matplotlib.pyplot as plt

import IPID_1

############################
### Stuart-Landau system ###
############################

# select parameters
mu = 1. # >0
eta = 1.
alpha = -0.4

# isostable characteristics
omega_SL = eta - alpha*mu
kappa_SL = -2.*mu

def PRC_Stuart_Landau(phi):
    return -1./np.sqrt(mu)*(np.sin(phi)+alpha*np.cos(phi))

def IRC_Stuart_Landau(phi):
    return 2./np.sqrt(mu)*np.cos(phi)

def dx(state, I):
    x = state[0]
    y = state[1]
    return mu*x - eta*y - (x**2+y**2)*(x-alpha*y) + I 

def dy(state, I):
    x = state[0]
    y = state[1]
    return mu*y + eta*x - (x**2+y**2)*(y+alpha*x)

def one_step_integrator_RK4(state, dstate, I, dt):
	"""RK4 integrates state with derivative for one step of dt
	
	:param state: state of the variables
	:param dstate: derivative functions
	:param dt: time step
    :param I: external stimulation
	:return: state after one integration step"""
	D = len(state)
	# 1
	k1 = [dstate[i](state, I) for i in range(D)]
	# 2
	state2 = [state[i]+k1[i]*dt/2.0 for i in range(D)]
	k2 = [dstate[i](state2, I) for i in range(D)]
	# 3
	state3 = [state[i]+k2[i]*dt/2.0 for i in range(D)] 
	k3 = [dstate[i](state3, I) for i in range(D)]
	# 4
	state4 = [state[i]+k3[i]*dt for i in range(D)] 
	k4 = [dstate[i](state4, I) for i in range(D)]
	# put together
	statef = [state[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0*dt for i in range(D)]
	return statef

#####################
### generate data ###
#####################

# stimulation parameters
T = 8. # period of pulses
I = 1. # pulse amplitude
tau = 0.05 # pulse duration 

# integration parameters
dt = 0.001 # time step
N = 300000 # number of time steps

Time = dt*np.arange(N)

# initialize "states"
states = np.zeros((2,N))
states[:,0] = np.array([1.,0.]) 

# initialize "stimulation"
stimulation  = np.zeros(N)
stimulation[np.mod(Time,T) < tau] = I

# integrate Stuart-Landau
for i in range(N-1):
    states[:,i+1] = one_step_integrator_RK4(states[:,i], [dx,dy], stimulation[i], dt)

##############
### IPID-1 ###
##############

# we select "x" as the observable
threshold = 0.

omega_infer, PRC_modes, phase = IPID_1.infer_response_curves(states[0], stimulation, dt, 
                                threshold=threshold,
                                N_Fourier=8,
                                iterations=5)


###############
### results ###
###############

phi = np.linspace(0., 2.*np.pi, 300)

# display state space
plt.plot(states[0], states[1], label='trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.axvline(threshold, ls='--', c='b', alpha=0.5, label='threshold')
plt.axis('equal')
plt.title('state space')
plt.legend()
plt.show()

#display time series
plt.title('time evolution')
plt.plot(Time, states[0], c='b', label='x')
plt.plot(Time, stimulation, c='k', label='stimulation')
plt.axhline(threshold, ls='--', c='b', alpha=0.5, label='threshold')
plt.legend()
plt.show()

# display PRC
plt.plot(phi, PRC_Stuart_Landau(phi), label='true PRC')
plt.plot(phi, IPID_1.f_from_modes_array(PRC_modes[-1], samples=len(phi)), ls='--', label='inferred PRC')
plt.title('inference of phase dynamics')
plt.plot([], ls='', label='true freq. ='     + str(np.round(omega_SL, 4)))
plt.plot([], ls='', label='inferred freq. =' + str(np.round(omega_infer[-1], 4)))

plt.legend()
plt.show()

# display IRC
plt.plot(phi, IRC_Stuart_Landau(phi), label='true IRC')
plt.title('inference of isostable dynamics')
plt.legend()
plt.show()