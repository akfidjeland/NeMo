#ifndef NEMO_CONFIGURATION_IMPL_HPP
#define NEMO_CONFIGURATION_IMPL_HPP

//! \file ConfigurationImpl.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ostream>
#include <boost/optional.hpp>

#include <nemo/config.h>
#include "StdpFunction.hpp"
#include "constants.h"

#ifdef INCLUDE_MPI

#include <boost/serialization/serialization.hpp>

namespace boost {
	namespace serialization {
		class access;
	}
}

#endif

namespace nemo {

class NEMO_BASE_DLL_PUBLIC ConfigurationImpl
{
	public:

		ConfigurationImpl();

		/*! Switch on logging and send output to stdout */
		void enableLogging() { m_logging = true; }

		void disableLogging() { m_logging = false; }
		bool loggingEnabled() const { return m_logging; }

		void setCpuThreadCount(unsigned threads);
		unsigned cpuThreadCount() const { return m_cpuThreadCount; }

		void setCudaPartitionSize(unsigned ps) { m_cudaPartitionSize = ps; }
		unsigned cudaPartitionSize() const { return m_cudaPartitionSize; }

		/*! Set the size of the firing buffer such that it can contain a fixed
		 * number of \a cycles worth of firing data before overflowing. */
		void setCudaFiringBufferLength(unsigned cycles) { m_cudaFiringBufferLength = cycles; }
		unsigned cudaFiringBufferLength() const { return m_cudaFiringBufferLength; }

		void setCudaDevice(unsigned device) { m_cudaDevice = device; }
		unsigned cudaDevice() const { return m_cudaDevice; }

		void setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		const boost::optional<StdpFunction>& stdpFunction() const { return m_stdpFn; }

		void setFractionalBits(unsigned bits);

		/*! \return the number of fractional bits. If the user has not
		 * specified this (\see fractionalBitsSet) the return value is
		 * undefined */
		unsigned fractionalBits() const;

		bool fractionalBitsSet() const;

		void setBackend(backend_t backend);
		backend_t backend() const { return m_backend; }

		void setBackendDescription(const std::string& descr) { m_backendDescription = descr; }
		const std::string& backendDescription() const { return m_backendDescription; }

	private:

		bool m_logging;
		boost::optional<StdpFunction> m_stdpFn;

		int m_fractionalBits;
		static const int s_defaultFractionalBits = -1;

		/* CPU-specific */
		unsigned m_cpuThreadCount;

		/* CUDA-specific */
		unsigned m_cudaPartitionSize;
		unsigned m_cudaFiringBufferLength; // in cycles

		unsigned m_cudaDevice;

		friend void check_close(const ConfigurationImpl& lhs, const ConfigurationImpl& rhs);

		backend_t m_backend;

		std::string m_backendDescription;

#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & m_logging;
			ar & m_stdpFn;
			ar & m_fractionalBits;
			ar & m_cpuThreadCount;
			ar & m_cudaPartitionSize;
			ar & m_cudaFiringBufferLength;
			ar & m_backend;
			ar & m_backendDescription;
		}
#endif
};


}


NEMO_BASE_DLL_PUBLIC
std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf);

#endif
