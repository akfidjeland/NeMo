#include "Mapper.hpp"

namespace nemo {
	namespace mpi {


Mapper::Mapper(int workers) :
	m_workers(workers)
{
	;
}


int
Mapper::rankOf(nidx_t n) const
{
	//! \todo do something sensible here. Use a size hint.
	//! \todo make sure we're not using more nodes than we have available
	return 1 + n / 1024;
}

	} // end namespace mpi
} // end namespace nemo
