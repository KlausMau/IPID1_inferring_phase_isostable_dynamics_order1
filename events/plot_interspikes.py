from matplotlib import pyplot

f = open("events.txt","rt")
lines = f.readlines()
f.close()
events = [float(line) for line in lines]
devents = [events[i+1]-events[i] for i in range(len(events)-1)]

f = open("../forcing/forcing.txt","rt")
lines = f.readlines()
f.close()
forcing = [float(line) for line in lines]
forcing_at_events = [300*(forcing[int(event)-1]+forcing[int(event)]+forcing[int(event)+1]) for event in events]

f = open("../phase/phase.txt","rt")
lines = f.readlines()
f.close()
phase = [float(line) for line in lines]
phase_at_events = [100*phase[int(event)] for event in events]

f = open("../signal/signal_x.txt","rt")
lines = f.readlines()
f.close()
signal = [float(line) for line in lines]
signal_at_events = [3000*signal[int(event)] for event in events]

pyplot.plot(devents)
pyplot.plot(forcing_at_events)
pyplot.plot(phase_at_events)
pyplot.plot(signal_at_events)
pyplot.show()

pyplot.plot(signal)
for event in events:
	pyplot.plot([event,event],[0,1],'k',alpha=0.2)
pyplot.show()
