#include "Mapper.hpp"

namespace nemo {
	namespace mpi {


Mapper::Mapper(int workers, int rank) :
	//! \todo do something sensible here. Use a size hint.
	//! \todo support heterogenous clusters
	m_nodeSize(512),
	m_workers(workers),
	m_rank(rank),
	m_startIdx(m_rank * m_nodeSize)
{
	;
}


int
Mapper::rankOf(nidx_t n) const
{
	//! \todo make sure we're not using more nodes than we have available
	return 1 + n / m_nodeSize;
}


nidx_t
Mapper::localIndex(nidx_t global) const
{
	global - m_startIdx;
}



unsigned
Mapper::localCount() const
{
	return m_nodeSize;
}


	} // end namespace mpi
} // end namespace nemo
