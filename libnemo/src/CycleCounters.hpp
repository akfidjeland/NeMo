#ifndef CYCLE_COUNTERS_HPP
#define CYCLE_COUNTERS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

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
