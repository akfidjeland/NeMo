#ifndef CONNECTIVITY_MATRIX_HPP
#define CONNECTIVITY_MATRIX_HPP

#include <vector>
#include <map>
#include <set>

#include "types.h"

namespace nemo {
	namespace cpu {

struct Synapse
{
	Synapse(weight_t w, nidx_t t) : weight(w), target(t) {}

	weight_t weight; 
	nidx_t target; 
};


struct ForwardIdx
{
	ForwardIdx(nidx_t source, delay_t delay) : source(source), delay(delay) {}

	nidx_t source;
	delay_t delay;
};



/* A row contains number of synapses with a fixed source and delay */
struct Row
{
	Row() : len(0), data(NULL) {}

	Row(Synapse* data, size_t len) : data(data), len(len) {}

	size_t len;
	Synapse* data; // freed in ConnectivityMatrix destructor
};



class ConnectivityMatrix
{
	public:

		ConnectivityMatrix();

		~ConnectivityMatrix();

		/*! Add synapses for a particular presynaptic neuron and a particular delay */
		void setRow(
				nidx_t source,
				delay_t delay,
				const nidx_t* targets,
				const weight_t* weights,
				size_t length);

		const Row& getRow(nidx_t source, delay_t) const;

		void finalize();

		delay_t maxDelay() const { return m_maxDelay; }

	private:

		/* Need to deal with manually allocated 'row' memory if we want to copy
		 * the whole CM around, so just don't */
		ConnectivityMatrix(const ConnectivityMatrix&);
		ConnectivityMatrix& operator=(const ConnectivityMatrix&);

		/* During network construction we accumulate data in a map. This way we
		 * don't need to know the number of neurons or the number of delays in
		 * advance */
		std::map<ForwardIdx, Row> m_acc;

		/* At run-time however, we want the fastest possible lookup of the
		 * rows. We therefore use a vector with linear addressing. This just
		 * points to the data in the accumulator. This is constructed in \a
		 * finalize which must be called prior to getRow being called */
		std::vector<Row> m_cm;

		bool m_finalized;

		std::set<nidx_t> m_sourceIndices;
		delay_t m_maxDelay;

		/*! \return linear index into CM, based on 2D index (neuron,delay) */
		size_t addressOf(nidx_t, delay_t) const;
};



inline
size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	return source * m_maxDelay + delay - 1;
}


}
}


#endif
