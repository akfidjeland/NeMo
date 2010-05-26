#ifndef NEMO_EXAMPLES_HPP
#define NEMO_EXAMPLES_HPP

namespace nemo {

	/* Construct a network with ncount neurons each of which has scount
	 * synapses. The synapses are given uniformly random targets from the whole
	 * population. 80% of synapses are excitatory and 20% are inhibitory, with
	 * weights chosen as in Izhikevich' reference implementation. All delays
	 * are 1ms.*/
	namespace random1k {
		nemo::Network* construct(unsigned ncount, unsigned scount);
	}

	namespace torus {
		nemo::Network* construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging);
	}
}

#endif
