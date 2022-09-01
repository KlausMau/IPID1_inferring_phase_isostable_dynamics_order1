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
signal_y = []
f = open("signal/signal_y.txt", "rt")
for line in f.readlines():
	signal_y.append(float(line[0:-1]))
f.close()
# import forcing
forcing = []
f = open("forcing/forcing.txt", "rt")
for line in f.readlines():
	forcing.append(float(line[0:-1])*parameters.forcing_strength)
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

# calculate prc
period = eq_phase_rec.oscillator_period(parameters.ders)
prc = eq_phase_rec.oscillator_PRC(parameters.ders, [0,1,0], period, thr=parameters.threshold, initial_warmup_periods=15, warmup_periods=5)
# calculate arc
floquet = eq_phase_rec.oscillator_floquet(parameters.ders, period)
arc = eq_phase_rec.oscillator_ARC(parameters.ders, [0,1,0], period, floquet, thr=parameters.threshold, initial_warmup_periods=25)

# figure
fig = pyplot.figure(figsize=(13, 3))
ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2)
ax3 = fig.add_subplot(1,4,3)
ax4 = fig.add_subplot(1,4,4)

# true
ax1.plot(prc[0][0:-1], prc[1][0:-1], lw=6, c='k', alpha=0.5)
# iterations
ax1.plot(prc[0], [PRC_from_sol(ph, prc_iterations[0])*parameters.prc_scale for ph in prc[0]], c='b', alpha=0.5)
ax1.plot(prc[0], [PRC_from_sol(ph, prc_iterations[1])*parameters.prc_scale for ph in prc[0]], c='g', alpha=0.5)
ax1.plot(prc[0], [PRC_from_sol(ph, prc_iterations[2])*parameters.prc_scale for ph in prc[0]], c='#AA6600', alpha=0.5)
# final
ax1.plot(prc[0], [PRC_from_sol(ph, sol_prc)*parameters.prc_scale for ph in prc[0]], c='r')
ax1.plot([0,2*pi],[0,0],'k',alpha=0.3)
ax1.set_xlabel(r"$\varphi$")
ax1.set_ylabel(r"$Z(\varphi)$")

# true
ax2.plot(arc[0], arc[1], lw=6, c='k', alpha=0.5)
# iterations
ax2.plot(arc[0], [PRC_from_sol(ph, arc_iterations[0])*parameters.arc_scale for ph in arc[0]], c='b', alpha=0.5)
ax2.plot(arc[0], [PRC_from_sol(ph, arc_iterations[1])*parameters.arc_scale for ph in arc[0]], c='g', alpha=0.5)
ax2.plot(arc[0], [PRC_from_sol(ph, arc_iterations[2])*parameters.arc_scale for ph in arc[0]], c='#AA6600', alpha=0.5)
# final
ax2.plot(arc[0], [PRC_from_sol(ph, sol_arc)*parameters.arc_scale for ph in arc[0]], c='r')
ax2.plot([0,2*pi],[0,0],'k',alpha=0.3)
ax2.set_xlabel(r"$\varphi$")
ax2.set_ylabel(r"$I(\varphi)$")

ax3.plot(signal_x[1000:4000], c='#1155EE')
ax3.plot(forcing[1000:4000], c='g')
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$x$")

ax4.plot(signal_x[1000:11000:5], signal_y[1000:11000:5], lw=0.5, c='#1155EE')
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$y$")

pyplot.tight_layout(pad=2.0)
pyplot.savefig("figure.pdf")
pyplot.show()

