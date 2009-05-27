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

		unsigned long long* dataLTP() const { return m_ccLTP.deviceData(); }
		size_t pitchLTP() const { return m_ccLTP.wordPitch(); }

		unsigned long long* dataLTD() const { return m_ccLTD.deviceData(); }
		size_t pitchLTD() const { return m_ccLTD.wordPitch(); }

		unsigned long long* dataConstrain() const { return m_ccConstrain.deviceData(); }
		size_t pitchConstrain() const { return m_ccConstrain.wordPitch(); }

	private:

		//! \todo use a single list of counters (but with different sizes)
		NVector<unsigned long long> m_ccMain;
		NVector<unsigned long long> m_ccLTP;
		NVector<unsigned long long> m_ccLTD;
		NVector<unsigned long long> m_ccConstrain;

		size_t m_partitionCount;

		int m_clockRateKHz;

		bool m_stdpEnabled;

		void printCounterSet(
				NVector<unsigned long long>& cc_in,
				size_t counters,
				const char* setName,
				const char* names[], // for intermediate counters
				std::ofstream& outfile);
};

#endif
