#ifndef NEMO_MAPPER_HPP
#define NEMO_MAPPER_HPP

namespace nemo {

/*! Mapping between neuron index spaces
 *
 * The mapping is between a global index space (G) and a local index space (L).
 * Concrete classes are specific to a backend.
 *
 * The mapper may have a valid range of global neurons indices which it deals
 * with. Additionally the mapper is informed of the exact set of neurons which
 * are in use (through \a addGlobal). An arbitrary neuron index can thus be
 * classified as either 'valid' (in the range) or 'existing' (i.e. already
 * specified).
 *
 * \see nemo::cpu::Mapper
 * \see nemo::cuda::Mapper
 */
template<class G, class L>
class Mapper
{
	public :

		virtual ~Mapper() { }

		/*! Translate from global to local index */
		virtual L localIdx(const G&) const = 0;

		/*! Translate from local to global index */
		virtual G globalIdx(const L&) const = 0;

		/*! Add a neuron to the set of existing neurons and return the local
		 * index. */
		virtual L addGlobal(const G&) = 0;

		/*! \return
		 * 		true if the global neuron index is one which has previously
		 * 		been added to the mapper (via \a addGlobal), false otherwise.
		 */
		virtual bool existingGlobal(const G&) const = 0;

		/*! \return
		 * 		true if the local neuron index is on which has previously been
		 * 		added to the mapper (via \a addGlobal), false otherwise
		 */
		virtual bool existingLocal(const L&) const = 0;

		virtual L maxLocalIdx() const = 0;

		/*! \return
		 * 		number of neurons in the valid index range. This may be larger
		 * 		than the existing number of neurons in this range as some
		 * 		indices may be unused.*/
		virtual unsigned neuronsInValidRange() const = 0;
};

}

#endif
