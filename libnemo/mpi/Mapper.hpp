#ifndef NEMO_MPI_MAPPER_HPP
#define NEMO_MPI_MAPPER_HPP

#include <nemo/internal_types.h>

namespace nemo {
	namespace mpi {

/* Translate between global neuron indices and rank indices
 *
 * Each neuron is processed on a single node. The index of a neuron can thus be
 * specified either in a global index or with a rank/local index pair.
 */
class Mapper
{
	public:

		/*! Create a new mapper.
		 *
		 * \param neurons total number of neurons in the network
		 * \param workers total number of workers in MPI workgroup
		 */
		Mapper(unsigned neurons, unsigned workers, int rank);

		/*! \return the rank of the process which should process a particular neuron */ 
		int rankOf(nidx_t) const;

		/*! \return number of neurons handled locally */
		unsigned neuronCount() const;

	private:

		/* Number of neurons processed on each node */
		int m_nodeSize;
};


	} // end namespace mpi
} // end namespace nemo

#endif
