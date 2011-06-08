import sys

import nemo
import random

net = nemo.Network()

# Excitatory neurons
for nidx in range(800):
    r = random.random()**2
    c = -65.0+15*r
    d = 8.0 - 6.0*r
    net.add_neuron(nidx, 0.02, 0.2, c, d, 5.0, 0.2*c, c)
    targets = range(1000)
    weights = [0.5*random.random() for tgt in targets]
    net.add_synapse(nidx, targets, 1, weights, False)

# Inhibitory neurons
for nidx in range(800,1000):
    nidx = 800 + n
    r = random.random()
    a = 0.02+0.08*r
    b = 0.25-0.05*r
    c = -65.0
    net.add_neuron(nidx, a, b, c, 2.0, 2.0, b*c, c)
    targets = range(1000)
    weights = [-random.random() for tgt in targets]
    net.add_synapse(nidx, targets, 1, weights, False)

conf = nemo.Configuration()
sim = nemo.Simulation(net, conf)
for t in range(1000):
    fired = sim.step()
    print t, ":", fired
