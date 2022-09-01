import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # perspective from the main directory ../

from math import pi, floor, ceil

# import events
events = []
f = open("events/events.txt", "rt")
for line in f.readlines():
	events.append(float(line[0:-1]))
f.close()

# write phase
f = open("phase/phase.txt", "wt")
for t in range(floor(events[0])):
	f.write("-1\n")
for i in range(len(events)-1):
	T = events[i+1]-events[i]
	for t in range(ceil(events[i]),floor(events[i+1])+1):
		f.write(str(2*pi*(t-events[i])/T)+"\n")
f.write("-1\n")
f.close()
