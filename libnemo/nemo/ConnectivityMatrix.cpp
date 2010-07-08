/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConnectivityMatrix.hpp"

#include <algorithm>
#include <utility>
#include <stdlib.h>
#include <boost/tuple/tuple_comparison.hpp>

#include <nemo/config.h>
#include "exception.hpp"
#include "fixedpoint.hpp"


namespace nemo {


Row::Row(const std::vector<AxonTerminal<nidx_t, weight_t> >& ss, unsigned fbits) :
	len(ss.size())
{
	FAxonTerminal<fix_t>* ptr;
#ifdef HAVE_POSIX_MEMALIGN
	//! \todo factor out the memory aligned allocation
	int error = posix_memalign((void**)&ptr,
			ASSUMED_CACHE_LINE_SIZE,
			ss.size()*sizeof(FAxonTerminal<fix_t>));
	if(error) {
		throw nemo::exception(NEMO_ALLOCATION_ERROR, "Failed to allocate CM row");
	}
#else
	ptr = (FAxonTerminal<fix_t>*) malloc(ss.size()*sizeof(FAxonTerminal<fix_t>));
#endif

	data = boost::shared_array< FAxonTerminal<fix_t> >(ptr, free);

	/* static/plastic flag is not needed in forward matrix */
	for(std::vector<nemo::AxonTerminal<nidx_t, weight_t> >::const_iterator si = ss.begin();
			si != ss.end(); ++si) {
		size_t i = si - ss.begin();
		ptr[i] = FAxonTerminal<fix_t>(fx_toFix(si->weight, fbits), si->target);
	}
}



ConnectivityMatrix::ConnectivityMatrix(unsigned fractionalBits) :
	m_fractionalBits(fractionalBits),
	m_maxDelay(0),
	m_maxSourceIdx(0)
{
	;
}



Row&
ConnectivityMatrix::setRow(nidx_t source,
		delay_t delay,
		const std::vector<AxonTerminal<nidx_t, weight_t> >& ss)
{
	std::pair<std::map<fidx, Row>::iterator, bool> insertion =
		m_acc.insert(std::make_pair<fidx, Row>(fidx(source, delay), Row(ss, m_fractionalBits)));
	if(!insertion.second) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Double insertion into connectivity matrix");
	}
	m_delays[source].insert(delay);
	m_maxSourceIdx = std::max(m_maxSourceIdx, source);
	m_maxDelay = std::max(m_maxDelay, delay);
	return insertion.first->second;
}



/* The fast lookup is indexed by source and delay. */
void
ConnectivityMatrix::finalizeForward()
{
	m_cm.resize((m_maxSourceIdx+1) * m_maxDelay);

	for(nidx_t n=0; n <= m_maxSourceIdx; ++n) {
		for(delay_t d=1; d <= m_maxDelay; ++d) {
			std::map<fidx, Row>::const_iterator row = m_acc.find(fidx(n, d));
			if(row != m_acc.end()) {
				m_cm.at(addressOf(n,d)) = row->second;
			} else {
				m_cm.at(addressOf(n,d)) = Row(); // defaults to empty row
			}
			//! \todo can delete the map now
		}
	}
}



const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	return m_cm.at(addressOf(source, delay));
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_begin(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_delays.find(source);
	if(found == m_delays.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.begin();
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_end(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_delays.find(source);
	if(found == m_delays.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.end();
}


} // namespace nemo
