#ifndef NEMO_MPI_MAPPER_HPP
#define NEMO_MPI_MAPPER_HPP

namespace nemo {
	namespace mpi {

#include <types.h>

class Mapper
{
	public:

		Mapper(int workers);

		/*! \return the rank of the process which should process a particular neuron */ 
		int rankOf(nidx_t) const;

	private:

		int m_workers;
};


	} // end namespace mpi
} // end namespace nemo

#endif
