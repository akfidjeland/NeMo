/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <boost/tuple/tuple_comparison.hpp>

#include "RSMatrix.hpp"
#include "connectivityMatrix.cu_h"
#include "device_memory.hpp"
#include "exception.hpp"


namespace nemo {
	namespace cuda {


RSMatrix::RSMatrix(size_t partitionSize) :
	mh_source(partitionSize),
	mh_sourceAddress(partitionSize),
	m_partitionSize(partitionSize),
	mw_pitch(0),
	mb_allocated(0)
{ }



/*! Allocate device memory and a linear block of host memory of the same size */
boost::shared_ptr<uint32_t>&
RSMatrix::allocateDeviceMemory()
{
	size_t desiredPitch = maxSynapsesPerNeuron() * sizeof(uint32_t);
	size_t height = RCM_SUBMATRICES * m_partitionSize;
	size_t bytePitch = 0;

	void* d_data = NULL;
	d_mallocPitch(&d_data, &bytePitch, desiredPitch, height, "rcm synapse group");
	md_data = boost::shared_ptr<uint32_t>(static_cast<uint32_t*>(d_data) , d_free);

	mw_pitch = bytePitch / sizeof(uint32_t);
	mb_allocated = bytePitch * height;

	d_memset2D((void*) md_data.get(), bytePitch, 0, height);

	return md_data;
}



bool
RSMatrix::onDevice() const
{
	return md_data.get() != NULL;
}



size_t
RSMatrix::planeSize() const
{
	return m_partitionSize * mw_pitch;
}



size_t
RSMatrix::maxSynapsesPerNeuron() const
{
	size_t n = 0;
	for(host_plane::const_iterator i = mh_source.begin();
			i != mh_source.end(); ++i) {
		n = std::max(n, i->size());
	}
	return n;
}


void
RSMatrix::moveToDevice(
		host_plane& h_mem,
		size_t plane,
		uint32_t defaultValue,
		uint32_t* d_mem)
{
	/* We only need to store the addresses on the host side */
	std::vector<uint32_t> buf(planeSize(), defaultValue);
	for(host_plane::const_iterator n = h_mem.begin(); n != h_mem.end(); ++n) {
		size_t offset = (n - h_mem.begin()) * mw_pitch;
		std::copy(n->begin(), n->end(), buf.begin() + offset);
	}
	h_mem.clear();
	memcpyToDevice(d_mem + plane * planeSize(), buf, planeSize());
}


void
RSMatrix::moveToDevice()
{
	boost::shared_ptr<uint32_t> d_mem = allocateDeviceMemory();
	moveToDevice(mh_source, RCM_ADDRESS, INVALID_REVERSE_SYNAPSE, d_mem.get());
	moveToDevice(mh_sourceAddress, RCM_FADDRESS, 0, d_mem.get());
}



void 
RSMatrix::addSynapse(
		const DeviceIdx& source,
		unsigned targetNeuron,
		unsigned delay,
		uint32_t forwardAddress)
{
	/*! \note we cannot check source partition or neuron here, since this class
	 * only deals with the reverse synapses for a single partition. It should
	 * be checked in the caller */
	uint32_t synapse = r_packSynapse(source.partition, source.neuron, delay);
	mh_source.at(targetNeuron).push_back(synapse);
	mh_sourceAddress.at(targetNeuron).push_back(forwardAddress);
}



void
RSMatrix::clearStdpAccumulator()
{
	//! \todo allocate data once in ctor instead, obvioating need for this check.
	if(!onDevice()) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"attempting to clear STDP array before device memory allocated");
	}
	d_memset2D(d_stdp(), mw_pitch*sizeof(uint32_t), 0, m_partitionSize);
}



uint32_t*
RSMatrix::d_address() const
{
	return md_data.get() + RCM_ADDRESS * planeSize();
}



uint32_t*
RSMatrix::d_faddress() const
{
	return md_data.get() + RCM_FADDRESS * planeSize();
}


weight_dt*
RSMatrix::d_stdp() const
{
	return (weight_dt*) md_data.get() + RCM_STDP * planeSize();
}

	} // end namespace cuda
} // end namespace nemo
