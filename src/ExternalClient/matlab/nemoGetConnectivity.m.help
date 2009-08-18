nemoGetConnectivity: get current connectivity matrix 
----------------------------------------------------

::

	[TARGETS, DELAYS, WEIGHTS] = nemoGetConnectivity()

The weights may change during simulation when using STDP. This function reads back the entire weight matrix, thus letting the user directly see the effect that STDP has (beyond a change of firing patterns).

The returned matrix has the same format as that used in ``nemoSetNetwork``, i.e. an N-by-M matrix, where N is the number of neurons and M is the maximum number of synapses per neuron.

Even if the synapses are static, the weights returned by this function may differ slightly from the input weights given to ``nemoSetNetwork``. This is due to different floating point formats on the backend.

The order of synapses in the returned matrix will almost certainly differ from the order in the input to ``nemoSetNetwork``. The connectivity matrix may be transformed in various ways as it's being mapped onto the backend. These transformations are not reversed. Every call to ``nemoGetConnectivity`` will return the synapses in the same order, though.

