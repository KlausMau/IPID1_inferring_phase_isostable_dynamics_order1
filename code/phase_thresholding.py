import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
import numpy as np
from matplotlib import pyplot

# import phase
phase = []
f = open("phase/phase.txt", "rt")
for line in f.readlines():
	phase.append(float(line[0:-1]))
f.close()
# import signal
signal = []
f = open("signal/signal_x.txt", "rt")
for line in f.readlines():
	signal.append(float(line[0:-1]))
f.close()

events = []
a_at_events = []
# thresholding
for i in range(len(phase)-1):
	pdif1 = phase[i]-parameters.phase_threshold
	pdif2 = phase[i+1]-parameters.phase_threshold
	if(pdif1*pdif2 < 0): # here there can be more than one event per period and thats ok
		if(pdif2 > 0):
			events.append(i+abs(pdif1)/(abs(pdif1)+abs(pdif2)))
			a_at_events.append(signal[i]+(signal[i+1]-signal[i])*abs(pdif1)/(abs(pdif1)+abs(pdif2)))
#pyplot.plot(signal)
#pyplot.plot(phase)
#for event in events:
#	pyplot.plot([event,event],[-5,5],'k')
#for i in range(len(events)):
#	pyplot.plot([events[i],events[i]],[0,a_at_events[i]],'r')
#pyplot.plot([0,len(signal)],[parameters.threshold,parameters.threshold])
#pyplot.plot([0,len(signal)],[parameters.phase_threshold,parameters.phase_threshold],'b')
#pyplot.show()

# write on file
f = open("events/phase_thr_events.txt", "wt")
for i in range(len(events)):
	f.write(str(events[i])+"\n")
f.close()
f = open("events/amplitude_x0_at_events.txt", "wt")
for i in range(len(a_at_events)):
	f.write(str(a_at_events[i])+"\n")
f.close()
