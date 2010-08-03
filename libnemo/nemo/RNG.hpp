#ifndef NEMO_RNG_HPP
#define NEMO_RNG_HPP

#include <vector>
#include <nemo/types.h>
#include <nemo/config.h>

namespace nemo {

class NEMO_BASE_DLL_PUBLIC RNG
{

	public:

		RNG();

		RNG(unsigned seeds[]);

		float gaussian();

		unsigned uniform();

		unsigned operator[](size_t i) { return m_state[i]; }

	private:

		unsigned m_state[4];
};


/* Generates RNG seeds for neurons in the range [0, maxIdx], and writes the
 * seeds for [minIdx, maxIdx] to the output vector. Generating and discarding
 * the initial seed values is done in order to always have a fixed mapping from
 * global neuron index to RNG seed values. */
NEMO_BASE_DLL_PUBLIC
void
initialiseRng(nidx_t minNeuronIdx, nidx_t maxNeuronIdx, std::vector<RNG>& rngs);

} // end namespace

#endif
