#!/usr/bin/env python

import unittest
import nemo

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



if __name__ == '__main__':
    unittest.main()
