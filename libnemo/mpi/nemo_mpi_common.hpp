#ifndef NEMO_MPI_COMMON_HPP
#define NEMO_MPI_COMMON_HPP

namespace nemo {
	namespace mpi {

enum Ranks {
	MASTER = 0
};

enum CommTag {
	NEURON_SCALAR,
	SYNAPSE_VECTOR,
	END_CONSTRUCTION,
	MASTER_STEP,
	WORKER_STEP
};

	}
}

#endif
