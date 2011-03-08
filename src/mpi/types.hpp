#ifndef NEMO_MPI_TYPES_HPP
#define NEMO_MPI_TYPES_HPP

#include <boost/serialization/vector.hpp>
#include <nemo/types.hpp>


namespace nemo {
	namespace mpi {

/* Every cycle the master synchronises with each worker. */
class SimulationStep
{
	public :

		SimulationStep() :
			terminate(false) { }

		SimulationStep(bool terminate, std::vector<unsigned> fstim):
			terminate(terminate), fstim(fstim) { }

		/* Add neuron to list of neurons which should be
		 * forced to fire */
		void forceFiring(nidx_t neuron) {
			fstim.push_back(neuron);
		}

		bool terminate;
		std::vector<unsigned> fstim;

	private :

		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & terminate;
			ar & fstim;
		}
};


	}
}

#endif
