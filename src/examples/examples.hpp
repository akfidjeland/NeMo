#ifndef NEMO_EXAMPLES_HPP
#define NEMO_EXAMPLES_HPP

namespace nemo {

	/* Construct a network with ncount neurons each of which has scount
	 * synapses. The synapses are given uniformly random targets from the whole
	 * population. 80% of synapses are excitatory and 20% are inhibitory, with
	 * weights chosen as in Izhikevich' reference implementation. All delays
	 * are 1ms.
	 *
	 * If 'stdp' is true, all excitatory synapses are marked as plastic, while
	 * inhibitory synapses are marked as static.
	 */
	namespace random {
		nemo::Network* construct(unsigned ncount, unsigned scount, bool stdp);
	}

	namespace torus {
		nemo::Network* construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging);
	}
}

#endif
