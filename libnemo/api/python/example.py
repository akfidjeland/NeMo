#!/usr/bin/env python

import random

import nemo


def add_excitatory(net, nidx, ncount, scount, stdp=False):
    v = -65.0
    a = 0.02
    b = 0.2
    # TODO: merge these?
    r1 = random.random() ** 2
    r2 = random.random() ** 2
    c = v + 15.0 * r1
    d = 8.0 - 6.0 * r2
    u = b * v
    sigma = 5.0
    net.add_neuron(nidx, a, b, c, d, u, v, sigma)
    for s in range(scount):
        target = random.randint(0, ncount-1)
        weight = 0.5 * random.random()
        net.add_synapse(nidx, target, 1, weight, stdp)
    return net


def add_inhibitory(net, nidx, ncount, scount):
    v = -65.0
    r1 = random.random()
    a = 0.02 + 0.08 * r1
    r2 = random.random()
    b = 0.25 - 0.05 * r2
    c = v
    d = 2.0
    u = b * v
    sigma = 2.0
    net.add_neuron(nidx, a, b, c, d, u, v, sigma)
    for s in range(scount):
        target = random.randint(0, ncount-1)
        weight = -random.random()
        net.add_synapse(nidx, target, 1, weight, False)
    return net


def construct_random(ncount, scount):
    """
    Construct a randomly connected network with n neurons each of which connect to m postsynaptic neurons.

    """
    def is_excitatory(nidx):
        return nidx < (ncount * 4 / 5)

    net = nemo.Network()
    for nidx in range(ncount):
        if is_excitatory(nidx):
            add_excitatory(net, nidx, ncount, scount)
        else:
            add_inhibitory(net, nidx, ncount, scount)
    return net


def run_random(ncount, scount, duration=1000):
    print "configure"
    conf = nemo.Configuration()
    print "construct"
    net = construct_random(ncount, scount)
    print "create simulation"
    sim = nemo.Simulation(net, conf)
    # TODO: factor out simulation runners in put in nemo.util
    print "run simulation"
    for t in range(duration):
        fired = sim.step()
        print t, ": ", fired


if __name__ == "__main__":
    run_random(1000, 1000) 
