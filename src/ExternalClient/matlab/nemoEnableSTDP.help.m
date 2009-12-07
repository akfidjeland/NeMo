nemoEnableSTDP: enable and configure STDP
-----------------------------------------

::

	nemoEnableSTDP(PRE_FIRE, POST_FIRE, MAX_WEIGHT, MIN_WEIGHT)

If set before the first call to ``nemoRun``, the backend is configured to run with STDP. It will gather synapse modification (LTP and LTD) statistics continously. To update the synapses using the accumulated values, call ``nemoApplySTDP``.

Synapses are modified either when a spike arrives shortly before or shortly after the postsynaptic neuron fires.

The vectors ``PRE_FIRE`` and ``POST_FIRE`` specify the value that is added to the synapse weight (when ``nemoApplySTDP`` is called) in the two cases for different values of dt+1 (where dt is time difference between spike arrival and firing). The +1 is due to the 1-based indexing used in Matlab; it's possible to have dt=0. For example ``PRE_FIRE[2]`` specifies the term to add to a synapse for which a spike arrived one cycle before the postsynaptic fired.

The length of each vector specify the time window during which STDP has an effect.

In the regular asymetric STDP, ``PRE_FIRE`` leads to potentiation and is hence positive, whereas ``POST_FIRE`` leads to depression and is hence negative.  However, the STDP function can be configured differently. Any cycle for which the function has a positive value is a potentation cycle, and conversely, negative values indicate depression. This results in two distinct regions of the STDP window. These regions are treated differently. For any postsynaptic firing only the spike whose arrival is closest in time to the firing is used to update the potentation/depression statistics.

Only excitatory synapses are affectd by STDP. The weights of the affected synapses *excitatory* synapses are never allowed to increase above ``MAX_WEIGHT``, and likewise not allowed to go negative.  Conversely, *inhibitory* synapses are never allowed to decrease below ``MIN_WEIGHT`` or to go positive. However, in for both excitatory and inhibitory synapses it is possible for a synapse to recover after reaching weight 0. 

STDP is disabled by default. When ``nemoEnableSTDP`` is called it is enabled for all subsequent simulation until ``nemoDisableSTDP`` is called. 

