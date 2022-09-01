import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
from math import pi, sin, cos, ceil, floor, sqrt
from matplotlib import pyplot
import numpy as np

# run in c
#os.system("g++ -I ./code/ code/arc_infer.c -o code/arc_infer.out")
os.system("./code/arc_infer.out "+str(parameters.prc_N_fourier))

"""
def ARC_from_sol(fi, sol):
	res = float(sol[1])
	for n in range(1, parameters.prc_N_fourier):
		res += float(sol[2*n])*cos(n*fi)
		res += float(sol[2*n+1])*sin(n*fi)
	return res

# import events
events = []
f = open("events/phase_thr_events.txt", "rt")
for line in f.readlines():
	events.append(float(line[0:-1]))
f.close()
events = events[:-1] # one less to avoid out of range
# import amplitude at events
a_at_events = []
f = open("events/amplitude_x0_at_events.txt", "rt")
for line in f.readlines():
	a_at_events.append(float(line[0:-1]))
f.close()
a_at_events = a_at_events[:-1] # one less to avoid out of range
# import amplitude
amplitude = []
f = open("amplitude/amplitude_x0.txt", "rt")
for line in f.readlines():
	amplitude.append(float(line[0:-1]))
f.close()
# import phase
phase = []
f = open("phase/phase.txt", "rt")
for line in f.readlines():
	phase.append(float(line[0:-1]))
f.close()
# import forcing
forcing = []
f = open("forcing/forcing.txt", "rt")
for line in f.readlines():
	forcing.append(float(line[0:-1]))
f.close()

A = [[] for i in range(len(events)-1)]
b = np.matrix([a_at_events[i+1]-a_at_events[i] for i in range(len(events)-1)]).transpose()
for i in range(len(events)-1):
	integral = amplitude[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
	for t in range(ceil(events[i]),floor(events[i+1])):
		integral += amplitude[t]
	integral += amplitude[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
	A[i].append(-integral) # floquet exponent
	# fourier integrals
	integral = forcing[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
	for t in range(ceil(events[i]),floor(events[i+1])):
		integral += forcing[t]
	integral += forcing[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
	A[i].append(integral)
	for n in range(1,parameters.prc_N_fourier):
		# cos
		integral = forcing[ceil(events[i])-1]*cos(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
		for t in range(ceil(events[i]),floor(events[i+1])):
			integral += forcing[t]*cos(n*phase[t])
		integral += forcing[floor(events[i+1])]*cos(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
		A[i].append(integral)
		# sin
		integral = forcing[ceil(events[i])-1]*sin(n*phase[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
		for t in range(ceil(events[i]),floor(events[i+1])):
			integral += forcing[t]*sin(n*phase[t])
		integral += forcing[floor(events[i+1])]*sin(n*phase[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
		A[i].append(integral)
	# the k*x0 integral
	integral_x0 = ceil(events[i])-events[i] # first fractional timestep
	for t in range(ceil(events[i]),floor(events[i+1])):
		integral_x0 += 1
	integral_x0 += events[i+1]-floor(events[i+1]) # last fractional timestep
	A[i].append(integral_x0) # floquet exponent time x0

# minimization
A = np.matrix(A)
AT = A.transpose()
ATA = AT*A
ATAinv = np.linalg.inv(ATA)
sol = ATAinv*(AT*b)

# write on file
f = open("ARC/sol.txt", "wt")
for i in range(len(sol)):
	f.write(str(float(sol[i]))+"\n")
f.close()

# plot
#args = [0.1*i for i in range(int(2*pi/0.1))]
#arc = [ARC_from_sol(arg,sol) for arg in args]
#pyplot.plot(args,arc)
#pyplot.show()

# estimate x0
x0 = float(sol[-1])/float(sol[0]) # k*x0/k
#print("x0 = "+str(x0))

# amplitude recalculation
new_amplitude = []
psis = []
for t in range(floor(events[0])+1):
	new_amplitude.append(-1)
for i in range(len(events)-1):
	amp = a_at_events[i]-x0 + (-sol[0]*(a_at_events[i]-x0)+ARC_from_sol(0,sol)*forcing[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
	new_amplitude.append(amp)
	for t in range(ceil(events[i]),floor(events[i+1])):
		amp = amp + (-sol[0]*amp+ARC_from_sol(phase[t],sol)*forcing[t])
		new_amplitude.append(amp)
	# amplitude at the end
	psi = amp + (-sol[0]*amp+ARC_from_sol(phase[t],sol)*forcing[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
	psis.append(psi)
	# rescale so its a_at_events[i+1]-x0 at the end
	# TURNED THE RESCALLING OFF, I THINK IT MAKES A BETTER FIT
	#for t in range(ceil(events[i]),floor(events[i+1])+1):
		#new_amplitude[t] = new_amplitude[t]*((1-(t-events[i])/(events[i+1]-events[i]))+(t-events[i])/(events[i+1]-events[i])*(a_at_events[i+1]-x0)/psi)

# write on file
f = open("amplitude/amplitude.txt", "wt")
for i in range(len(new_amplitude)):
	f.write(str(float(new_amplitude[i]))+"\n")
f.close()
f = open("amplitude/amplitude_x0.txt", "wt")
for i in range(len(new_amplitude)):
	f.write(str(float(new_amplitude[i])+x0)+"\n")
f.close()

# plot
#pyplot.plot([new_amplitude[i] for i in range(len(new_amplitude))])
#pyplot.plot(forcing,'g')
#for i in range(len(events)):
#	pyplot.plot([events[i],events[i]],[0,a_at_events[i]-x0],'k')
#pyplot.show()

# error
diffs = [(psis[i]-(a_at_events[i+1]-x0))**2 for i in range(len(psis))]
error = sqrt(np.average(diffs))
# 0 error
a_sqs = [(a_at_event-x0)**2 for a_at_event in a_at_events]
error0 = sqrt(np.average(a_sqs))

# write on file
f = open("error/error_arc.txt", "wt")
f.write(str(error)+"\n")
f.close()
f = open("error/error0_arc.txt", "wt")
f.write(str(error0)+"\n")
f.close()
"""
