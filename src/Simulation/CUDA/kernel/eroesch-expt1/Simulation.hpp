#ifndef SIMULATION_H
#define SIMULATION_H

//! \file Simulation.hpp

#include "Cluster.hpp"
#include "izhikevich.h"
#include <vector>

/*! \brief Structure containing all the information required for simulation on CUDA devices 
 *
 * The external API is c-based, so typically Simulation objects are manipulated
 * through the functions prototyped in izhikevich.h
 * 
 * \author Andreas Fidjeland
 */
class Simulation {

	public :

		Simulation();

		/*! \see nsimAddCluster */
		int addCluster(const Cluster&);

		/*! \see nsimAddCluster */
		int addCluster(int n, 
				const float* v,
				const float* u,
				const float* a,
				const float* b,
				const float* c,
				const float* d, 
				const float* connectionStrength,
				bool hasExternalCurrent=false,
				bool hasExternalFiring=false);

		/*! 
		 * \param outputStart
		 * 		First simulation cycle for which output should be generated.
		 * \param outputDuration
		 * 		Number of simulation cycles for which output should be
		 * 		generate. If set to 0 all output untill the end is generated.
		 */
		SimStatus run(int simCycles, 
				int updatesPerInvocation,
				int reportFlags, 
				float currentScaling,
				void(*currentStimulus)(float*, int, int),
				void(*firingStimulus)(char*, int, int),
				HandlerError(*firingHandler)(FILE*, char*, int, int, uchar),
				FILE* firingFile,
				HandlerError(*vHandler)(FILE*, float*, int),
				FILE* vFile,
				int outputStart=0,
				int outputDuration=0,
				bool forceDense=false);

		/* call setProbe to specify which cluster to inspect. At the moment
		 * only a single cluster is supported.
		 * \todo support larger nets, and parts of clusters */
		void setClusterProbe(int clusterIndex);

		/* specify specific neurons to inspect. Ad the moment only all or a
		 * single neuron is supported */
		void setNeuronProbe(int neuronIndex);

		int maxClusterSize() const;

	private :

		std::vector<Cluster> clusters;

		int m_clusterProbe;

		int m_neuronProbe;

		enum NeuronProbe {
			ALL = -1	
		};
};


#endif
