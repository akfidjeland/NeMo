#include "Mapper.hpp"

namespace nemo {
	namespace cpu {

Mapper::Mapper(const nemo::NetworkImpl& net) :
	m_offset(0)
{
	if(net.neuronCount() > 0) {
		m_offset = net.minNeuronIndex();
	}
}

/* Convert global neuron index to local */
nidx_t
Mapper::localIdx(nidx_t global) {
	return global - m_offset;
}
		
		/* Convert local neuron index to global */
		nidx_t globalIdx(nidx_t local) {
			return local + m_offset;
		}

	private :

		nidx_t m_offset;
};

}	}
