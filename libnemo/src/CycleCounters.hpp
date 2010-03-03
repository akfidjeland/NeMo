#ifndef CYCLE_COUNTERS_HPP
#define CYCLE_COUNTERS_HPP

#include "NVector.hpp"

namespace nemo {

struct CycleCounters
{
	public:

		CycleCounters(size_t partitionCount, bool stdpEnabled);

		void printCounters(std::ostream& out);

		unsigned long long* data() const;

		/*! \return word pitch for cycle counting arrays */
		size_t pitch() const;

		unsigned long long* dataApplySTDP() const { return m_ccApplySTDP.deviceData(); }
		size_t pitchApplySTDP() const { return m_ccApplySTDP.wordPitch(); }

	private:

		//! \todo use a single list of counters (but with different sizes)
		NVector<unsigned long long> m_ccMain;
		NVector<unsigned long long> m_ccApplySTDP;

		size_t m_partitionCount;

		unsigned long long m_clockRateKHz;

		bool m_stdpEnabled;

		void printCounterSet(
				NVector<unsigned long long>& cc_in,
				size_t counters,
				const char* setName,
				const char* names[], // for intermediate counters
				std::ostream& outfile);
};

} // end namespace nemo

#endif
