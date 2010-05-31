#ifndef NEMO_MPI_TYPES_HPP
#define NEMO_MPI_TYPES_HPP

#include <boost/serialization/vector.hpp>
#include <types.hpp>

namespace nemo {
	namespace mpi {

//! \todo might re-use this data-type in NetworkImpl.
/* On the wire, send synapses grouped by source and delay */
class SynapseVector
{
	public :

		//! \todo need to ensure this is the same type as used in NetworkImpl.
		//Define in terms of typedefs inside NetworkImpl.
		typedef nemo::AxonTerminal<nidx_t, weight_t> terminal_t;

		SynapseVector() :
			source(0), delay(0) { }

		SynapseVector(nidx_t source, delay_t delay,
				const std::vector<terminal_t>& terminals) :
			source(source), delay(delay), terminals(terminals) { }

		nidx_t source;
		delay_t delay;
		std::vector<terminal_t> terminals;	

	private :

		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & source;
			ar & delay;
			ar & terminals;
		}
};



/* Every cycle the master synchronises with each worker. */
class SimulationStep
{
	public :

		SimulationStep() :
			terminate(false) { }

		SimulationStep(bool terminate, std::vector<unsigned> fstim):
			terminate(terminate), fstim(fstim) { }

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
