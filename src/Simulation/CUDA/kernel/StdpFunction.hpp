#ifndef STDP_FUNCTION_HPP
#define STDP_FUNCTION_HPP

//! \file StdpFunction.hpp

#include <stdint.h>
#include <vector>

/*! \brief User-configurable STDP function */
class StdpFunction
{
	public:

		StdpFunction(unsigned int preFireWindow,
				unsigned int postFireWindow,
				uint64_t potentiationBits,
				uint64_t depressionBits,
				float* stdpFn,
				float maxWeight,
				float minWeight);

		void configureDevice();

		float maxWeight() const;

		float minWeight() const;

	private:

		std::vector<float> m_function;

		unsigned int m_preFireWindow;
		unsigned int m_postFireWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		float m_maxWeight;
		float m_minWeight;
};

#endif
