nemoSetNetwork: send a complete network to nemo
-----------------------------------------------

::

	nemoSetNetwork(A, B, C, D, U, V, TARGETS, DELAYS, WEIGHTS)
	nemoSetNetwork(A, B, C, D, U, V, TARGETS, DELAYS, WEIGHTS, PLASTIC)

The neuron population is defined by ``A``-``D``, ``U``, and ``V`` which are all N-by-1 matrices, where N is the number of neurons in the network.  

The connectivity is specified using the three or four N-by-M matrices ``TARGETS``, ``WEIGHTS``, ``DELAYS``, and optionally ``PLASTIC``. N is again the number of neurons in the network and M is the maximum number of synapses per neuron. If a neuron has less than M outgoing synapses, invalid synapses should point to neuron 0. ``PLASTIC`` is a logical matrix specifying for each synapse whether it's plastic (i.e. can change at run-time) or not. If this argument is left out all synapses are assumed to be static. 

The simulation is discrete-time, so delays are rounded to integer values.  

A connection to nemo must have already been set up using ``nemoConnect``.

