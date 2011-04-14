/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Neurons.hpp"

#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {

Neurons::Neurons(const NeuronType& type) :
	mf_param(type.f_nParam()),
	mf_state(type.f_nState()),
	m_size(0),
	m_type(type)
{
	;
}


size_t
Neurons::add(const float fParam[], const float fState[])
{
	for(unsigned i=0; i < mf_param.size(); ++i) {
		mf_param[i].push_back(fParam[i]);
	}
	for(unsigned i=0; i < mf_state.size(); ++i) {
		mf_state[i].push_back(fState[i]);
	}
	return m_size++;
}



unsigned
Neurons::parameterIndex(unsigned i) const
{
	using boost::format;
	if(i >= mf_param.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid parameter index %u") % i));
	}
	return i;
}



unsigned
Neurons::stateIndex(unsigned i) const
{
	using boost::format;
	if(i >= mf_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid state variable index %u") % i));
	}
	return i;
}



void
Neurons::set(size_t n, const float fParam[], const float fState[])
{
	for(unsigned i=0; i < mf_param.size(); ++i) {
		mf_param[i][n] = fParam[i];
	}
	for(unsigned i=0; i < mf_state.size(); ++i) {
		mf_state[i][n] = fState[i];
	}
}



float
Neurons::getParameter(size_t nidx, unsigned pidx) const
{
	return mf_param[parameterIndex(pidx)][nidx];
}



float
Neurons::getState(size_t nidx, unsigned sidx) const
{
	return mf_state[stateIndex(sidx)][nidx];
}



float
Neurons::getMembranePotential(size_t nidx) const
{
	return getState(nidx, m_type.membranePotential());
}



void
Neurons::setParameter(size_t nidx, unsigned pidx, float val)
{
	mf_param[parameterIndex(pidx)][nidx] = val;
}


void
Neurons::setState(size_t nidx, unsigned sidx, float val)
{
	mf_state[stateIndex(sidx)][nidx] = val;
}


}
