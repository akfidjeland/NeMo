//! \file StdpFunction.cpp

#include <algorithm>

#include "StdpFunction.hpp"


StdpFunction::StdpFunction(unsigned int preFireWindow,
		unsigned int postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		float* stdpFn,
		float maxWeight):
	m_function(postFireWindow + preFireWindow, 0.0f),
	m_preFireWindow(preFireWindow),	
	m_postFireWindow(postFireWindow),
	m_potentiationBits(potentiationBits),
	m_depressionBits(depressionBits),
	m_maxWeight(maxWeight)
{
	std::copy(stdpFn, stdpFn + m_function.size(), m_function.begin());
}


extern void
configureStdp(
		uint preFireWindow,
		uint postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		float* stdpFn);


void
StdpFunction::configureDevice()
{
	configureStdp(m_preFireWindow, 
			m_postFireWindow, 
			m_potentiationBits,
			m_depressionBits,
			&m_function[0]);
}



float
StdpFunction::maxWeight() const
{
	return m_maxWeight;
}
