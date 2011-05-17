#ifndef NEMO_PYTHON_NETWORK_HPP
#define NEMO_PYTHON_NETWORK_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <nemo/Network.hpp>

namespace nemo {
	namespace python {
		
/* Network class with Python-specific extension
 *
 * This class mainly deals with 'kwargs'/variadic functions.
 */
class Network : public nemo::Network
{
	public :

		Network() : m_args(16, 0.0f) { }

		/*! Add one ore more neurons of arbitrary type
		 *
		 * This function expects the following arguments:
		 *
		 * - neuron type (unsigned)
		 * - neuron index (unsigned)
		 * - a variable number of parameters (float)
		 * - a variable number of state variables (float)
		 *
		 * The corresponding python prototype would be 
		 *
		 * add_neuron(self, neuron_type, neuron_idx, *args)
		 *
		 * The neuron type is expected to be a scalar. The remaining arguments
		 * may be either scalar or vector. All vectors must be of the same
		 * length. If any of the inputs are vectors, the scalar arguments are
		 * replicated for each synapse.
		 *
		 * \param args parameters and state variables
		 * \param kwargs unused, but required by boost::python
		 * \return None
		 *
		 * \see addNeuronType
		 */
		boost::python::object addNeuronVarargs(
				boost::python::tuple args,
				boost::python::dict kwargs);

	private :

		/* Pre-allocate temporaries used for argument marshalling */

		/*! Parameters and state variables */
		std::vector<float> m_args;

		/*! All input arguments, before casting */
		std::vector<PyObject*> m_objects;

		/*! Vectorization status for each argument */
		std::vector<bool> m_vectorized;
};


/* Add a single neuron of arbitrary type
 *
 * \see nemo::python::addNeuronVarargs
 */
inline
boost::python::object
add_neuron_va(boost::python::tuple args, boost::python::dict kwargs)
{
	return boost::python::extract<nemo::python::Network&>(args[0])()
						.addNeuronVarargs(args, kwargs);
}




}	}


#endif
