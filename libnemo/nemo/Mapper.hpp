#ifndef NEMO_MAPPER_HPP
#define NEMO_MAPPER_HPP

namespace nemo {

/* Class which performs mapping from a global index space (G) to a local
 * index space (L).
 */

template<class G, class L>
class Mapper
{
	public :

		/* Translate from global to local index */
		virtual L localIdx(const G&) const = 0;

		/* Translate from local to global index */
		virtual G globalIdx(const L&) const = 0;

		/* Add a neuron to the set of valid neurons and return the local index. */
		virtual L addGlobal(const G&) = 0;

		virtual bool validGlobal(const G&) const = 0;

		virtual bool validLocal(const L&) const = 0;

		virtual L maxLocalIdx() const = 0;

		/*! \return number of valid neurons registered with the mapper */
		virtual unsigned neuronCount() const = 0;
};

}

#endif
