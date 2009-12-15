nemoRun: Perform one or more simulation steps
---------------------------------------------

::

	F = nemoRun(NSTEPS, FSTIM)

Returns an N-by-2 matrix containing the firing information for the next ``NSTEPS`` simulation steps on an already constructed network. 

Each row in the return matrix refers to one firing event, specifying the time and the neuron index. The time is in the range [1, ``NSTEPS``]. In other words the client code has to keep track of the total time elapsed, if this is of
interest.

The firing stimulus has the same format as the return matrix, and is used to specify neurons which should be forced to fire at some specific time. The firing stimulus should be sorted by cycle number. Out-of-bounds cycle values or neuron indices may lead to undefined behaviour.

The simulation is discrete-time, and any floating point values in ``FSTIM`` will be converted to ``uint32``. 
