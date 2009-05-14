//! \file Cluster.cpp

#include "Cluster.hpp"
#include "termcolour.hpp"

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>



Cluster::Cluster(int n) :
	n(n),
	m_v(n, 0),
	m_u(n, 0),
	m_a(n, 0),
	m_b(n, 0),
	m_c(n, 0),
	m_d(n, 0),
	m_connectionStrength(n*n, 0),
	m_connectionDelay(n*n, 0),
	m_hasExternalCurrent(false),
	m_hasExternalFiring(false),
	m_maxDelay(1)
{ }



Cluster::Cluster(int n, 
		const float* v,
		const float* u,
		const float* a,
		const float* b,
		const float* c,
		const float* d,
		const float* connectionStrength,
		const unsigned char* connectionDelay) :
	n(n),
	m_v(n, 0),
	m_u(n, 0),
	m_a(n, 0),
	m_b(n, 0),
	m_c(n, 0),
	m_d(n, 0),
	m_connectionStrength(n*n, 0),
	m_connectionDelay(n*n, 0),
	m_hasExternalCurrent(false),
	m_hasExternalFiring(false)
{
	std::copy(connectionStrength, connectionStrength+n*n,
			m_connectionStrength.begin());
	if(connectionDelay != NULL)
		std::copy(connectionDelay, connectionDelay+n*n,
				m_connectionDelay.begin());
	std::copy(v, v+n, m_v.begin());
	std::copy(u, u+n, m_u.begin());
	std::copy(a, a+n, m_a.begin());
	std::copy(b, b+n, m_b.begin());
	std::copy(c, c+n, m_c.begin());
	std::copy(d, d+n, m_d.begin());
}




//=============================================================================
// Neuron state 
//=============================================================================


void
Cluster::setV(int nn, float v)
{
	m_v[nn] = v;
}



void
Cluster::setU(int nn, float u)
{
	m_u[nn] = u;
}



void
Cluster::setA(int nn, float a)
{
	m_a[nn] = a;
}



void
Cluster::setB(int nn, float b)
{
	m_b[nn] = b;
}



void
Cluster::setC(int nn, float c)
{
	m_c[nn] = c;
}


void
Cluster::setD(int nn, float d)
{
	m_d[nn] = d;
}



//=============================================================================
// Connectivity 
//=============================================================================


const float*
Cluster::connectionStrength() const
{
	return &m_connectionStrength[0];
}



float
Cluster::connectionStrength(int pre, int post) const
{
	return m_connectionStrength[n*pre+post];
}



const unsigned char*
Cluster::connectionDelay() const
{
	return &m_connectionDelay[0];
}



unsigned char
Cluster::connectionDelay(int pre, int post) const
{
	return m_connectionDelay[n*pre+post];
}



unsigned char
Cluster::maxDelay() const
{
	return m_maxDelay;
}



void
Cluster::connect(int pre, int post, float strength, unsigned char delay)
{
	if(pre < 0 || pre >= n) {
		throw std::out_of_range("Cluster::connect: invalid presynaptic neuron");
	}
	if(post < 0 || post >= n) {
		throw std::out_of_range("Cluster::connect: invalid postsynaptic neuron");
	}

	//! \todo this does not deal correctly with double wiring, as we only store
	// a single delay.
	float existingStrength = m_connectionStrength[n*pre+post];	
	if(existingStrength != 0.0) {
		m_connectionStrength[n*pre+post] = strength + existingStrength;
	} else {
		m_connectionStrength[n*pre+post] = strength;
		m_connectionDelay[n*pre+post] = delay;
		m_maxDelay = std::max(m_maxDelay, delay);
	}
}



//! \todo may want to pre-compute/cache this
int
Cluster::postsynapticCount(int pre) const
{
	std::vector<float>::const_iterator b = m_connectionStrength.begin()+n*pre;
	std::vector<float>::const_iterator e = b + n;
	return n - std::count(b, e, 0.0f);
}



void
Cluster::disconnect(int pre, int post)
{
	if(pre < 0 || pre >= n) {
		throw std::out_of_range("Cluster::connect: invalid presynaptic neuron");
	}
	if(post < 0 || post >= n) {
		throw std::out_of_range("Cluster::connect: invalid postsynaptic neuron");
	}
	m_connectionStrength[n*pre+post] = 0.0f;
	m_connectionDelay[n*pre+post] = 0;
	//! \todo if m_maxDelay == delay we should reset m_maxDelay
}



std::vector<int>
Cluster::postIndices(int pre)
{
	if(pre < 0 || pre >= n) {
		throw std::out_of_range("Cluster::postIndices: invalid presynaptic neuron");
	}

	//! \todo use std::algorithm
	std::vector<int> ret;	
	const float* row = &m_connectionStrength[0] + pre*n;
	for( int i=0; i<n; ++i ){
		if(row[i] != 0.0f) {
			ret.push_back(i);
		}
	}

	return ret;
}



//=============================================================================
// Cluster configuration
//=============================================================================


void
Cluster::enableExternalCurrent()
{
	m_hasExternalCurrent = true;
}



bool 
Cluster::hasExternalCurrent() const
{
	return m_hasExternalCurrent;
}



void
Cluster::enableExternalFiring()
{
	m_hasExternalFiring = true;
}



bool
Cluster::hasExternalFiring() const
{
	return m_hasExternalFiring;
}



//=============================================================================
// Cluster statistics
//=============================================================================


float
Cluster::occupancy() const
{
	//! \todo operate on delay matrix instead
	int unused = std::count(m_connectionStrength.begin(),
			m_connectionStrength.end(), 0.0f); 
	int total = n*n;
	int occupied = total - unused;
	return float(occupied)/float(total);
}



int
Cluster::maxRowEntries() const
{
	int maxEntries = 0;
	std::vector<float>::const_iterator b = m_connectionStrength.begin();
	std::vector<float>::const_iterator e = b + n;
	for(int row=0; row<n; ++row) {
		int unused = std::count(b, e, 0.0f);
		maxEntries = std::max(maxEntries, n-unused);
		b += n;
		e += n;
	}
	return maxEntries;
}



//=============================================================================
// Visualisation
//=============================================================================


void
Cluster::printConnectivity() const
{
	int cw = columnWidth()-2;
	const int cs = n/cw + (n % cw ? 1 : 0); // cluster size
	cw = n/cs;

	for(int pre=0; pre<n; pre+=cs) {

		std::vector<bool> excitatory(cw, false);
		std::vector<bool> inhibitory(cw, false);

		for(int pre2=pre; pre2<pre+cs && pre2<n; ++pre2) {
			for(int post=0; post<n; ++post) {
				float s = connectionStrength(pre2, post);
				if(s < 0) {
					inhibitory[post/cs] = true;
				} else if(s > 0) {
					excitatory[post/cs] = true;
				}
			}
		}

		printf("|");
		for(int post=0; post<cw; ++post) {

			bool e = excitatory[post];	
			bool i = inhibitory[post];
			int colour;

			if(!e && !i) {
				colour = BLACK;
			} else if(e && !i) {
				colour = RED;
			} else if(!e && i) {
				colour = BLUE;
			} else {
				colour = MAGENTA;
			}

			setTextColour(stdout, BRIGHT, WHITE, colour);
			printf(" ");
		}
		setTextColour(stdout, RESET, WHITE, BLACK);
		printf("|\n");
	}

	//setTextColour(stdout, RESET, WHITE, BLACK);
}



//=============================================================================
// Related non-members
//=============================================================================

bool
operator<(const Cluster& lhs, const Cluster& rhs)
{
	return lhs.n < rhs.n;
}
