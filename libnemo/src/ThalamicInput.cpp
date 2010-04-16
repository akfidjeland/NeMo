//! \file ThalamicInput.cpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ThalamicInput.hpp"
#include "thalamicInput.cu_h"

#include <boost/random.hpp>


namespace nemo {
	namespace cuda {

ThalamicInput::ThalamicInput(
        size_t partitionCount,
        size_t partitionSize,
        int seed) :
    /*! \note For the RNG state we only use one state variable per thread (per
     * plane), rather than per neuron. We could thus make m_rngState smaller,
     * but make it the same size so that we can use the same pitch as the other
     * data structures. */
    m_rngState(partitionCount, partitionSize, true, 4),
    m_sigma(partitionCount, partitionSize, true, 1),
    m_inUse(false),
    m_partitionCount(partitionCount),
    m_partitionSize(partitionSize),
    m_seed(seed)
{ }



void
ThalamicInput::setNeuronSigma(size_t partition, size_t neuron, float val)
{
	m_inUse = val != 0.0f;
	m_sigma.setNeuron(partition, neuron, val);
}



size_t
ThalamicInput::d_allocated() const
{
	return m_rngState.d_allocated() + m_sigma.d_allocated();
}



void
ThalamicInput::moveToDevice()
{
    if(m_inUse) {
        initRngState();
        m_rngState.moveToDevice();
        m_sigma.moveToDevice();
    }
}



void
ThalamicInput::initRngState()
{
	typedef boost::mt19937 rng_t;

	//! \todo use seed here
	rng_t rng;

	boost::variate_generator<rng_t, boost::uniform_int<unsigned long> >
		seed(rng, boost::uniform_int<unsigned long>(0, 0x7fffffff));

    /* This RNG state vector needs to be filled with initialisation data.  Each
     * RNG needs 4 32-bit words of seed data, with each thread having a
     * diferent seed. */
    std::vector<unsigned> rngbuf(m_partitionSize);
    for(unsigned partition=0; partition<m_partitionCount; ++partition) {
        for(unsigned plane=0; plane<4; ++plane) {
            for(unsigned i=0; i<rngbuf.size(); ++i) {
				rngbuf[i] = seed();
            }
            m_rngState.setPartition(partition, &rngbuf[0], rngbuf.size(), plane);
        }
    }
}



unsigned* 
ThalamicInput::deviceRngState() const
{
    return m_inUse ? m_rngState.deviceData() : NULL;
}


float* 
ThalamicInput::deviceSigma() const
{
    return m_inUse ? m_sigma.deviceData() : NULL;
}


size_t
ThalamicInput::wordPitch() const
{
    size_t p1 = m_rngState.wordPitch();
    size_t p2 = m_sigma.wordPitch();
    //! \todo throw exception here instead
    assert(p1 == p2);
    return p1 == p2 ? p1 : 0;
}

	} // end namespace cuda
} // end namespace nemo
