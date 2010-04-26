#ifndef NEMO_EXAMPLES_HPP
#define NEMO_EXAMPLES_HPP

namespace nemo {
	namespace random1k {
		nemo::Network* construct(unsigned ncount);
	}
	namespace torus {
		nemo::Network* construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging);
	}
}

#endif
