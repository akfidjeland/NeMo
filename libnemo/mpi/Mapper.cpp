#include "Mapper.hpp"

#include <cassert>

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
	m_nodeSize(nodeSize(neurons, workers)),
	m_workers(workers),
	m_rank(rank),
	m_startIdx((m_rank - 1) * m_nodeSize)
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
Mapper::localIdx(const nidx_t& global) const
{
	assert(rankOf(global) == m_rank);
	assert(global >= m_startIdx);
	assert(global < m_startIdx + m_nodeSize);
	return global - m_startIdx;
}



nidx_t
Mapper::globalIdx(const nidx_t& local) const
{
	return m_startIdx + local;
}



unsigned
Mapper::localCount() const
{
	return m_nodeSize;
}



nidx_t
Mapper::maxLocalIdx() const
{
	return m_nodeSize - 1;
}



nidx_t
Mapper::addGlobal(const nidx_t& global)
{
	m_validGlobal.insert(global);
	return localIdx(global);

}



bool
Mapper::validGlobal(const nidx_t& global) const
{
	return m_validGlobal.count(global) == 1;
}


bool
Mapper::validLocal(const nidx_t& local) const
{
	return validGlobal(globalIdx(local));
}


	} // end namespace mpi
} // end namespace nemo
