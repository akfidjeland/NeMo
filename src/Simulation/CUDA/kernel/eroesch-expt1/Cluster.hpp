#ifndef CLUSTER_HPP
#define CLUSTER_HPP

//! \file Cluster.hpp

#include <vector>

/*! \brief Fully connected cluster of neurons
 *
 * \todo provide higher-level interface for setting parameters 
 * \todo use std::vector instead of raw c arrays
 *
 * \author Andreas Fidjeland
 */
class Cluster
{
	public :

		/*! Allocate memory for a cluster with \a n neurons */ 
		Cluster(int n);

		Cluster(int n, 
				const float* v,
				const float* u,
				const float* a,
				const float* b,
				const float* c,
				const float* d,
				const float* connectionStrength,
				const unsigned char* connectionDelay=NULL);
		
		//! \todo just make DeviceMemory a friend of this class to clean the interface
		const float* v() const { return &m_v[0]; }
		void setV(int neuron, float value);

		const float* u() const { return &m_u[0]; }
		void setU(int neuron, float value);

		const float* a() const { return &m_a[0]; }
		void setA(int neuron, float value);

		const float* b() const { return &m_b[0]; }
		void setB(int neuron, float value);

		const float* c() const { return &m_c[0]; }
		void setC(int neuron, float value);

		const float* d() const { return &m_d[0]; }
		void setD(int neuron, float value);


		/*! \return raw array of connection strength (densely packed n*n matrix */
		const float* connectionStrength() const;

		/*! \return connection strength between pre and post */
		float connectionStrength(int pre, int post) const;

		const unsigned char* connectionDelay() const;

		unsigned char connectionDelay(int pre, int post) const;

		unsigned char maxDelay() const;

		void connect(int pre, int post, float strength, unsigned char delay=1);

		void disconnect(int pre, int post);

		/*! \return the number of postsynaptic neurons connected to a
		 * presynaptic neuron. */
		int postsynapticCount(int pre) const;


		/*! \return vector of indices of postsynaptic neurons */
		std::vector<int> postIndices(int pre);

		/*! \return proportion of connections which are non-0 */
		float occupancy() const;

		/*! \return maximum number of non-0 entries among all rows */
		int maxRowEntries() const;

		/*! An external current is a current which is driven externally to the
		 * simulation. Not all clusters requires this and by default this is
		 * not enabled. Calling enableExternalCurrent switches this on for this
		 * cluster */
		void enableExternalCurrent();

		bool hasExternalCurrent() const;

		void enableExternalFiring();

		bool hasExternalFiring() const;

		void enableDelays();
		
		/*! Print conectivity to terminal */
		void printConnectivity() const;

		/* number of neurons */
		//! \todo make private
		int n;    

	private :

		/* Neuron parameters */
		std::vector<float> m_v;
		std::vector<float> m_u;
		std::vector<float> m_a;
		std::vector<float> m_b;
		std::vector<float> m_c;
		std::vector<float> m_d;

		/* Dense connectivity matrix, size n*n */
		std::vector<float> m_connectionStrength;

		/* Dense connectivity matrix, size n*n */
		std::vector<unsigned char> m_connectionDelay;

		bool m_hasExternalCurrent;

		bool m_hasExternalFiring;

		unsigned char m_maxDelay;
};


/* Related non-members */

bool operator<(const Cluster& , const Cluster&);

#endif
