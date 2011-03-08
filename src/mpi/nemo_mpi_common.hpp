#ifndef NEMO_MPI_COMMON_HPP
#define NEMO_MPI_COMMON_HPP

namespace nemo {
	namespace mpi {

enum Ranks {
	MASTER = 0
};

enum CommTag {
	NEURON_VECTOR,
	NEURONS_END,
	SYNAPSE_VECTOR,
	SYNAPSES_END,
	MASTER_STEP,
	WORKER_STEP
};

	}
}

#endif
