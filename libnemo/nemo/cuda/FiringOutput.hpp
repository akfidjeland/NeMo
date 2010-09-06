#ifndef FIRING_OUTPUT_HPP
#define FIRING_OUTPUT_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_ptr.hpp>

#include <nemo/internal_types.h>
#include "Mapper.hpp"

//! \file FiringOutput.hpp

/*! \brief Data and functions for reading firing data from device to host
 *
 * \author Andreas Fidjeland
 */

namespace nemo {
	namespace cuda {

class FiringOutput {

	public:

		/*! Set up data on both host and device for probing firing */
		FiringOutput(const Mapper& mapper);

		/*! Read firing data from device to host buffer. This should be called
		 * every simulation cycle */
		void sync();

		/*!  Read all firing data buffered on the device since the previous
		 * call to this function (or the start of simulation if this is the
		 * first call). The return vectors are valid until the next call to
		 * this function.
		 *
		 * \return
		 * 		Total number of cycles for which we return firing. The caller
		 * 		would most likely already know what this should be, so can use
		 * 		this for sanity checking.
		 */
		unsigned readFiring(
				const std::vector<unsigned>** cycles,
				const std::vector<unsigned>** neuronIdx);

		/*! Discard any locally buffered firing data */
		void flushBuffer();

		/*! \return device pointer to the firing buffer */
		uint32_t* d_buffer() const { return md_buffer.get(); }

		/*! \return bytes of allocated device memory */
		size_t d_allocated() const { return md_allocated; }

		size_t wordPitch() const { return m_pitch; }

	private:

		/* Dense firing buffers on device and host */
		boost::shared_ptr<uint32_t> md_buffer;
		boost::shared_ptr<uint32_t> mh_buffer; // pinned, same size as device buffer
		size_t m_pitch; // in words

		/* Internal host-side buffers which are updated every simulation cycle */
		std::vector<unsigned> m_cycles;
		std::vector<unsigned> m_neuronIdx;

		/* Host-side buffers exposed to the user. These are valid from one call
		 * to \a readFiring to the next */
		std::vector<unsigned> m_cyclesOut;
		std::vector<unsigned> m_neuronIdxOut;

		/*! \see step() */
		unsigned m_bufferedCycles;

		size_t md_allocated;

		void populateSparse(
				unsigned cycle,
				const uint32_t* hostBuffer,
				std::vector<unsigned>& firingCycle,
				std::vector<unsigned>& neuronIdx);

		Mapper m_mapper;
};

	} // end namespace cuda
} // end namespace nemo

#endif
