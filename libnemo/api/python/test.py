#!/usr/bin/env python

import unittest
import random
import nemo

def randomSource():
    return random.randint(0, 999)

def randomTarget():
    return randomSource()

def randomDelay():
    return random.randint(1, 20)

def randomWeight():
    return random.uniform(-1.0, 1.0)

def randomPlastic():
    return random.choice([True, False])


class TestFunctions(unittest.TestCase):


    def test_network_set_neuron(self):
        """ create a simple network and make sure we can get and set parameters
        and state variables """
        a = 0.02
        b = 0.2
        c = -65.0+15.0*0.25
        d = 8.0-6.0*0.25
        v = -65.0
        u = b * v
        sigma = 5.0

        net = nemo.Network()

        # This should only succeed for existing neurons
        self.assertRaises(RuntimeError, net.set_neuron, 0, a, b, c, d, u, v, sigma)

        net.add_neuron(0, a, b, c-0.1, d, u, v-1.0, sigma)

        # Getters should fail if given invalid neuron or parameter
        self.assertRaises(RuntimeError, net.get_neuron_parameter, 1, 0) # neuron
        self.assertRaises(RuntimeError, net.get_neuron_state, 1, 0)     # neuron
        self.assertRaises(RuntimeError, net.get_neuron_parameter, 0, 5) # parameter
        self.assertRaises(RuntimeError, net.get_neuron_state, 0, 2)     # state

        e = 0.1

        # Test setting whole neuron, reading back by parts
        net.set_neuron(0, a-e, b-e, c-e, d-e, u-e, v-e, sigma-e)

        # Since Python uses double precision and NeMo uses single precision
        # internally, the parameters may not be exactly the same after reading
        # back.
        
        places = 5
        self.assertAlmostEqual(net.get_neuron_parameter(0, 0), a-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 1), b-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 2), c-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 3), d-e, places)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 4), sigma-e, places)

        self.assertAlmostEqual(net.get_neuron_state(0, 0), u-e, places)
        self.assertAlmostEqual(net.get_neuron_state(0, 1), v-e, places)

        # Test setting and reading back neuron by parts

        net.set_neuron_parameter(0, 0, a)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 0), a, places)

        net.set_neuron_parameter(0, 1, b)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 1), b, places)

        net.set_neuron_parameter(0, 2, c)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 2), c, places)

        net.set_neuron_parameter(0, 3, d)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 3), d, places)

        net.set_neuron_parameter(0, 4, sigma)
        self.assertAlmostEqual(net.get_neuron_parameter(0, 4), sigma, places)

        net.set_neuron_state(0, 0, u)
        self.assertAlmostEqual(net.get_neuron_state(0, 0), u, places)

        net.set_neuron_state(0, 1, v)
        self.assertAlmostEqual(net.get_neuron_state(0, 1), v, places)

        # Individual setters should fail if given invalid neuron or parameter
        self.assertRaises(RuntimeError, net.set_neuron_parameter, 1, 0, 0.0) # neuron
        self.assertRaises(RuntimeError, net.set_neuron_state, 1, 0, 0.0)     # neuron
        self.assertRaises(RuntimeError, net.set_neuron_parameter, 0, 5, 0.0) # parameter
        self.assertRaises(RuntimeError, net.set_neuron_state, 0, 2, 0.0)     # state

    def test_add_synapse(self):
        """ The add_synapse method supports either vector or scalar input. This
        test calls set_synapse in a large number of ways, checking for
        catastrophics failures in the boost::python layer """

        def arg(vlen, gen):
            """ Return either a fixed-length vector or a scalar, with values
            drawn from 'gen'"""
            vector = random.choice([True, False])
            if vector:
                return [gen() for n in range(vlen)]
            else:
                return gen()

        net = nemo.Network()
        for test in range(1000):
            vlen = random.randint(2, 500)
            source = arg(vlen, randomSource)
            target = arg(vlen, randomTarget)
            delay = arg(vlen, randomDelay)
            weight = arg(vlen, randomWeight)
            plastic = arg(vlen, randomPlastic)
            net.add_synapse(source, target, delay, weight, plastic)


if __name__ == '__main__':
    unittest.main()
