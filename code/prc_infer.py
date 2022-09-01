import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
from math import pi, sin, cos, ceil, floor, sqrt
from matplotlib import pyplot
import numpy as np

# run in c
#os.system("g++ -I ./code/ code/prc_infer.c -o code/prc_infer.out")
os.system("./code/prc_infer.out "+str(parameters.prc_N_fourier))

"""
def PRC_from_sol(fi, sol):
	res = float(sol[1])
	for n in range(1, parameters.prc_N_fourier):
		res += float(sol[2*n])*cos(n*fi)
		res += float(sol[2*n+1])*sin(n*fi)
	return res

# import events
events = []
f = open("events/events.txt", "rt")
for line in f.readlines():
	events.append(float(line[0:-1]))
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
b = np.matrix([2*pi for i in range(len(events)-1)]).transpose()
for i in range(len(events)-1):
	A[i].append(events[i+1]-events[i]) # period
	# fourier integrals
	integral = forcing[ceil(events[i])-1]*(ceil(events[i])-events[i]) # first fractional timestep
	for t in range(ceil(events[i]),floor(events[i+1])):
		integral += forcing[t]
	integral += forcing[floor(events[i+1])]*(events[i+1]-floor(events[i+1])) # last fractional timestep
	A[i].append(integral) # natural frequency
	for n in range(1,parameters.prc_N_fourier):
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

# write on file
f = open("PRC/sol.txt", "wt")
for i in range(len(sol)):
	f.write(str(float(sol[i]))+"\n")
f.close()

# plot
#args = [0.1*i for i in range(int(2*pi/0.1))]
#prc = [PRC_from_sol(arg,sol) for arg in args]
#pyplot.plot(args,prc)
#pyplot.show()

# phase recalculation
new_phase = []
psis = []
for t in range(floor(events[0])+1):
	new_phase.append(-1)
for i in range(len(events)-1):
	ph = (sol[0]+PRC_from_sol(0,sol)*forcing[ceil(events[i])-1])*(ceil(events[i])-events[i]) # first fractional timestep
	new_phase.append(ph)
	for t in range(ceil(events[i]),floor(events[i+1])):
		ph = ph + sol[0] + PRC_from_sol(ph,sol)*forcing[t]
		new_phase.append(ph)
	# phase at the end
	psi = ph + (sol[0]+PRC_from_sol(ph,sol)*forcing[floor(events[i+1])])*(events[i+1]-floor(events[i+1])) # last fractional timestep
	psis.append(psi)
	# rescale so its 2pi at the end
	for t in range(ceil(events[i]),floor(events[i+1])+1):
		new_phase[t] = 2*pi*new_phase[t]/psi

# write on file
f = open("phase/phase.txt", "wt")
for i in range(len(new_phase)):
	f.write(str(float(new_phase[i]))+"\n")
f.close()

# plot
#pyplot.plot(new_phase)
#pyplot.plot(forcing)
#for event in events:
#	pyplot.plot([event,event],[0,2*pi],'k')
#pyplot.show()

# error
diffs = [(psi-2*pi)**2 for psi in psis]
error = sqrt(np.average(diffs))
# 0 error
Ts = [events[i+1]-events[i] for i in range(len(events)-1)]
avg_T = np.average(Ts)
avg_w = 2*pi/avg_T
diffs = [(avg_w*T-2*pi)**2 for T in Ts]
error0 = sqrt(np.average(diffs))

# write on file
f = open("error/error_prc.txt", "wt")
f.write(str(error)+"\n")
f.close()
f = open("error/error0_prc.txt", "wt")
f.write(str(error0)+"\n")
f.close()
"""
