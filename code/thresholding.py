import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

import parameters
from matplotlib import pyplot

# import signal
signal = []
f = open("signal/signal_x.txt", "rt")
for line in f.readlines():
	signal.append(float(line[0:-1]))
f.close()

events = []
# thresholding
for i in range(5,len(signal)-6):
	# four points, 5 below and 5 above threshold to count
	sdif1 = signal[i-5]-parameters.threshold
	sdif2 = signal[i]-parameters.threshold
	sdif3 = signal[i+1]-parameters.threshold
	sdif4 = signal[i+6]-parameters.threshold
	if(sdif2*sdif3 < 0 and sdif1*sdif4 < 0):
		if(sdif3 > 0 and sdif4 > 0):
			events.append(i+abs(sdif2)/(abs(sdif2)+abs(sdif3)))
#pyplot.plot(signal)
#for event in events:
#	pyplot.plot([event,event],[-5,5],'k')
#pyplot.plot([0,len(signal)],[parameters.threshold,parameters.threshold])
#pyplot.show()

# write on file
f = open("events/events.txt", "wt")
for i in range(len(events)):
	f.write(str(events[i])+"\n")
f.close()
