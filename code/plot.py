import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
from code import eq_phase_rec
from math import pi, sin, cos, sqrt
from matplotlib import pyplot

def PRC_from_sol(fi, sol):
	res = sol[1]
	for n in range(1, parameters.prc_N_fourier):
		res += sol[2*n]*cos(n*fi)
		res += sol[2*n+1]*sin(n*fi)
	return res

# import signal
signal_x = []
f = open("signal/signal_x.txt", "rt")
for line in f.readlines():
	signal_x.append(float(line[0:-1]))
f.close()
# import forcing
forcing = []
f = open("forcing/forcing.txt", "rt")
for line in f.readlines():
	forcing.append(float(line[0:-1]))
f.close()
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
# import prc solution
sol_prc = []
f = open("PRC/sol.txt", "rt")
for line in f.readlines():
	sol_prc.append(float(line[0:-1]))
f.close()
# import arc solution
sol_arc = []
f = open("ARC/sol.txt", "rt")
for line in f.readlines():
	sol_arc.append(float(line[0:-1]))
f.close()

# import prc and arc solution iterations
prc_iterations = []
arc_iterations = []
for n in range(1,5):
	prc_it = []
	f = open("PRC/iterations/"+str(n)+".txt", "rt")
	for line in f.readlines():
		prc_it.append(float(line[0:-1]))
	f.close()
	prc_iterations.append(prc_it)
	arc_it = []
	f = open("ARC/iterations/"+str(n)+".txt", "rt")
	for line in f.readlines():
		arc_it.append(float(line[0:-1]))
	f.close()
	arc_iterations.append(arc_it)

# figure
fig = pyplot.figure(figsize=(13, 3))
ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2)
ax3 = fig.add_subplot(1,4,3)
ax4 = fig.add_subplot(1,4,4)

# PRC
phase_args = [2*pi/100*i for i in range(100)]
# iterations
ax1.plot(phase_args, [PRC_from_sol(ph, prc_iterations[0])*parameters.prc_scale for ph in phase_args], c='b', alpha=0.5)
ax1.plot(phase_args, [PRC_from_sol(ph, prc_iterations[1])*parameters.prc_scale for ph in phase_args], c='g', alpha=0.5)
ax1.plot(phase_args, [PRC_from_sol(ph, prc_iterations[2])*parameters.prc_scale for ph in phase_args], c='#AA6600', alpha=0.5)
# final
ax1.plot(phase_args, [PRC_from_sol(ph, sol_prc)*parameters.prc_scale for ph in phase_args], c='r')
ax1.plot([0,2*pi],[0,0],'k',alpha=0.3)
ax1.set_xlabel(r"$\varphi$")
ax1.set_ylabel(r"$Z(\varphi)$")

# ARC
# iterations
ax2.plot(phase_args, [PRC_from_sol(ph, arc_iterations[0])*parameters.arc_scale for ph in phase_args], c='b', alpha=0.5)
ax2.plot(phase_args, [PRC_from_sol(ph, arc_iterations[1])*parameters.arc_scale for ph in phase_args], c='g', alpha=0.5)
ax2.plot(phase_args, [PRC_from_sol(ph, arc_iterations[2])*parameters.arc_scale for ph in phase_args], c='#AA6600', alpha=0.5)
# final
ax2.plot(phase_args, [PRC_from_sol(ph, sol_arc)*parameters.arc_scale for ph in phase_args], c='r')
ax2.plot([0,2*pi],[0,0],'k',alpha=0.3)
ax2.set_xlabel(r"$\varphi$")
ax2.set_ylabel(r"$I(\varphi)$")

ax3.plot(signal_x[1000:4000], c='#1155EE')
ax3.plot(forcing[1000:4000], c='g')
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$x$")

ax4.plot(signal_x[1000:11000:5], signal_x[1000+parameters.delay_emb:11000+parameters.delay_emb:5], lw=0.5, c='#1155EE')
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$y$")

pyplot.tight_layout(pad=2.0)
pyplot.savefig("figure.pdf")
pyplot.show()

