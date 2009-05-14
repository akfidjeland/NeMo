#ifndef CYCLE_COUNTERS_HPP
#define CYCLE_COUNTERS_HPP

#include "NVector.hpp"

struct CycleCounters
{
	public:

		CycleCounters(size_t partitionCount, int clockRateKHz);

		void printCounters(const char* outfile="cc.dat");

		unsigned long long* data() const;

		/*! \return word pitch for cycle counting arrays */
		size_t pitch() const;

	private:

		NVector<unsigned long long> m_cc;

		size_t m_partitionCount;

		int m_clockRateKHz;
};

#endif
