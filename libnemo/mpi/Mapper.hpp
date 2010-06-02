#ifndef NEMO_MPI_MAPPER_HPP
#define NEMO_MPI_MAPPER_HPP

namespace nemo {
	namespace mpi {

#include <types.h>

/* Translate between global and rank/local indices
 *
 * Each neuron is processed on a single node. The index of a neuron can thus be
 * specified either in a global index or with a rank/local index pair. This
 * class performs that mapping.
 */
class Mapper
{
	public:

		Mapper(int workers, int rank);

		/*! \return the rank of the process which should process a particular neuron */ 
		int rankOf(nidx_t) const;

		nidx_t localIndex(nidx_t global) const;

	private:

		/* Number of neurons processed on each node */
		int m_nodeSize;

		/* Total number of nodes */
		int m_workers;

		int m_rank;

		/* First index dealt with in *this* node */
		nidx_t m_startIdx;

};


	} // end namespace mpi
} // end namespace nemo

#endif
