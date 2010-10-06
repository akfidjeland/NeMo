#include "Mapper.hpp"

#include <nemo/util.h>
#include <nemo/exception.hpp>


namespace nemo {
	namespace mpi {


int
nodeSize(unsigned neurons, unsigned workers)
{
	if(workers == 0) {
		throw nemo::exception(NEMO_MPI_ERROR, "No worker nodes");
	}
	return DIV_CEIL(neurons, workers);
}



Mapper::Mapper(unsigned neurons, unsigned workers, int rank) :
	//! \todo support heterogenous clusters
	//! \todo leave nodes unused instead here, if nodes are not at capacity
	m_nodeSize(nodeSize(neurons, workers))
{
	;
}


int
Mapper::rankOf(nidx_t n) const
{
	//! \todo make sure we're not using more nodes than we have available
	return 1 + n / m_nodeSize;
}



unsigned
Mapper::neuronCount() const
{
	return m_nodeSize;
}



	} // end namespace mpi
} // end namespace nemo
