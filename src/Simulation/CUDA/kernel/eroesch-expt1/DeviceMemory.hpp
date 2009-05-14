#ifndef DEVICE_MEMORY_HPP
#define DEVICE_MEMORY_HPP

//! \file DeviceMemory.hpp

#include <vector>
#include <stdexcept>

/*! \brief Wrapper for the device memory used by simulation
 *
 * All pointers in this class are pointers to the device memory. Typically,
 * each structure will contain data for multiple clusters.
 *
 * \todo deal with hierarchical nets
 *
 * \author Andreas Fidjeland
 */
class DeviceMemory
{
	public :

		/* \param clusters
		 *		Vector of all the neuronal clusters to be used in this
		 *		simulation 
		 * \param forceDense 
		 * 		Force the connectivity matrix to use dense encoding
		 * \param verbose
		 * 		Print memory usage statistics to stdout */
		DeviceMemory(std::vector<class Cluster>& clusters, 
				bool forceDense=false,
				bool verbose=false);

		~DeviceMemory();

		/*! \return device address for firing array of a particular cluster. If
		 * the cluster index is invalid, return NULL. */
		int* firingAddress(int clusterIndex=0);

		/*! \return device address for v array of a particular cluster. If the
		 * cluster index is invalid, return NULL */
		float* vAddress(int clusterIndex);

		int clusterCount() const { return m_clusterCount; }

		//! \todo do the copying of constant parameters inside this class
		/*! \return pointer to device memory array with cluster configuration
		 * flags specifying whether cluster can receive external current. */
		const char* hasExternalCurrent() const;

		/*! \return pointer to device memory array with cluster configuration 
		 * flags specifying whether cluster firing can be driven from the host */
		const char* hasExternalFiring() const;

		/*! \return pointer to device memory array with cluster configuration
		 * specifying the maximum column index within each row when using
		 * sparse encoding */
		const int* maxColumnIndex() const;

		/*! \return true if *any* of the clusters use sparse encoding */
		bool sparseEncoding() const { return m_sparseEncoding; }

		unsigned char maxDelay() const;

	//! \todo make this private again
	//private :

		/* Neurons count */
		//! \todo set this appropriately, or dispense with it altogether
		int n;

		/* Neuron state */
		float* v;
		float* u;

		/* Neuron parameters */
		float* a;
		float* b;
		float* c;
		float* d;

		/* External input */
		float* extI;
		uint32_t* extFiring;

		/* Array of firing information for up to 32 most recent cycles for each
		 * neuron. LSb corresponds to most recent spike (delay=1), while MSb
		 * corresponds to the least recent one (delay=32).  */
		int* firing;

		/* Array of firing delays for each neuron. Follows same format is
		 * firing data. Each bit indicates whether there are *any* synapses
		 * leaving the neuron with the specified delay. The actual delay is
		 * found in the delays array or is packed along with the weights. */ 
		int* firingDelays;

		//! \todo make this sparse instead
		/* Dense connectivity matrix, size n*n */
		float* weights;

#ifndef BIT_PACK_DELAYS
		unsigned char* delays;
#endif

		size_t pitch32;
		size_t pitch8;
		size_t pitch1;

	private :

		int m_clusterCount;

		unsigned char m_maxDelay;

		/*! true if *any* cluster has sparse encoding */
		bool m_sparseEncoding;

		/*! print statistics of cluster configurations */
		bool m_verbose;

		std::vector<char> m_hasExternalCurrent;

		std::vector<char> m_hasExternalFiring;

		/*! Maximum column index for each cluster, for sparsely encoded
		 * connectivity */
		std::vector<int> m_maxColumnIndex;

		void allocateData(int clusterCount, int clusterSize);
		void copySparseConnectivity(int clusterIndex, const Cluster& cluster);
		void copyDenseConnectivity(int clusterIndex, const Cluster& cluster);
		void copyData(int clusterIndex, const Cluster& cluster, bool forceDense);
		void clearBuffers(int clusterCount);

		void setFiringDelays(const std::vector<Cluster>&);

		bool denseEncoding(const Cluster& cluster) const;
};




class cuda_mem_error : public std::runtime_error 
{
	public :

		cuda_mem_error(const std::string& msg = "") :
			std::runtime_error(msg) {}
};


#endif
