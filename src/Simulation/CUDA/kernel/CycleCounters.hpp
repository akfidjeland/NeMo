#ifndef CYCLE_COUNTERS_HPP
#define CYCLE_COUNTERS_HPP

#include "NVector.hpp"

struct CycleCounters
{
	public:

		//! \todo pass in the STDP option
		CycleCounters(size_t partitionCount, int clockRateKHz, bool stdpEnabled=true);

		void printCounters(const char* outfile="cc.dat");

		unsigned long long* data() const;

		/*! \return word pitch for cycle counting arrays */
		size_t pitch() const;

		unsigned long long* dataReorderSTDP() const { return m_ccReorderSTDP.deviceData(); }
		size_t pitchReorderSTDP() const { return m_ccReorderSTDP.wordPitch(); }

		unsigned long long* dataApplySTDP() const { return m_ccApplySTDP.deviceData(); }
		size_t pitchApplySTDP() const { return m_ccApplySTDP.wordPitch(); }

	private:

		//! \todo use a single list of counters (but with different sizes)
		NVector<unsigned long long> m_ccMain;
		NVector<unsigned long long> m_ccReorderSTDP;
		NVector<unsigned long long> m_ccApplySTDP;

		size_t m_partitionCount;

		unsigned long long m_clockRateKHz;

		bool m_stdpEnabled;

		void printCounterSet(
				NVector<unsigned long long>& cc_in,
				size_t counters,
				const char* setName,
				const char* names[], // for intermediate counters
				std::ofstream& outfile);
};

#endif
