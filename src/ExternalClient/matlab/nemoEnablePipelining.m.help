nemoEnablePipelining: increase throughtput by reducing latency 
--------------------------------------------------------------

::

	nemoEnablePipelining()

If set before the first simulation command, simulation will be run in a *pipelined* fashion. This can be used to increase the simulation throughput by hiding the network (that's the physical digital communication network, rather than the neural network) by overlapping communication with simulation. This introduces a delay between stimulus is given to the simulation and output can be read from it, and thus changes the behaviour of simulation commands. The length of this delay can be queried using ``nemoPipelineLength``.

