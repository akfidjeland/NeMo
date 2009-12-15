nemoStopSimulation
------------------

::

    nemoStopSimulation

Calling ``nemoStopSimulation`` terminates the simulation on whatever backend is currently in use, and frees up its resources. Following this function the network can be modified before further simulation takes place. ``nemoStopSimulation`` is called automatically when exiting matlab.

If there is no running simulation this function has no effect.
