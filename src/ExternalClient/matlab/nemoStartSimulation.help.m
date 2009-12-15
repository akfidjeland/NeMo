nemoStartSimulation
-------------------

::

    nemoStartSimulation

Calling ``nemoStartSimulation`` sets up the simulation so that it is ready to run. It is never strictly necessary to call this function, as the first call to any simulation function, such as ``nemoRun`` will set up the simulation if it has not already been done. This setup, however, can be quite expensive, so explicitly calling nemoStopSimulation may be desirable.

If the simulation is already running this function has no effect.
