import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

from math import pi, floor, ceil

# import events
events = []
f = open("events/phase_thr_events.txt", "rt")
for line in f.readlines():
	events.append(float(line[0:-1]))
f.close()
# import amplitudes at events
a_at_events = []
f = open("events/amplitude_x0_at_events.txt", "rt")
for line in f.readlines():
	a_at_events.append(float(line[0:-1]))
f.close()

# write amplitude
f = open("amplitude/amplitude_x0.txt", "wt")
for t in range(floor(events[0])):
	f.write("-1\n")
for i in range(len(events)-1):
	a0 = a_at_events[i]
	a1 = a_at_events[i+1]
	T = events[i+1]-events[i]
	for t in range(ceil(events[i]),floor(events[i+1])+1):
		f.write(str(a0+(t-events[i])*(a1-a0)/T)+"\n")
f.write("-1\n")
f.close()


