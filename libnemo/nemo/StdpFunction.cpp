#include "StdpFunction.hpp"
#include "exception.hpp"

namespace nemo {

StdpFunction::StdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight):
	m_prefire(prefire),
	m_postfire(postfire),
	m_minWeight(minWeight),
	m_maxWeight(maxWeight)
{ 
	if(minWeight > 0.0f) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP function should have a negative minimum weight");
	}

	if(maxWeight < 0.0f) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP function should have a positive maximum weight");
	}

	/*! \todo This constraint is too weak. Also need to consider max delay in
	 * network here */
	if(prefire.size() + postfire.size() > 64) {
		throw nemo::exception(NEMO_INVALID_INPUT, "size of STDP window too large");
	}

}



void
setBit(size_t bit, uint64_t& word)
{
	word = word | (uint64_t(1) << bit);
}



uint64_t
StdpFunction::getBits(bool (*pred)(float)) const
{
	uint64_t bits = 0;
	int n = 0;
	for(std::vector<float>::const_reverse_iterator f = m_postfire.rbegin();
			f != m_postfire.rend(); ++f, ++n) {
		if(pred(*f)) {
			setBit(n, bits);
		}
	}
	for(std::vector<float>::const_iterator f = m_prefire.begin(); 
			f != m_prefire.end(); ++f, ++n) {
		if(pred(*f)) {
			setBit(n, bits);
		}
	}
	return bits;
}


bool potentiation(float x ) { return x > 0.0f; }
bool depression(float x ) { return x < 0.0f; }


uint64_t
StdpFunction::potentiationBits() const
{
	return getBits(potentiation);
}


uint64_t
StdpFunction::depressionBits() const
{
	return getBits(depression);
}

}
