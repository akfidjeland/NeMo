/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Network.hpp"


bool checkInputVector(PyObject* obj, unsigned &vectorLength);

namespace nemo {
	namespace python {


boost::python::object
Network::addNeuronVarargs(
		boost::python::tuple py_args,
		boost::python::dict /*kw*/)
{
	using namespace boost::python;

	unsigned vlen = 0;
	unsigned nargs = boost::python::len(py_args);

	/* The neuron type should always be a scalar */
	unsigned neuron_type = extract<unsigned>(py_args[1]);

	/* Get raw pointers and determine the mix of scalar and vector arguments */
	//! \todo skip initial objects if possible
	m_objects.resize(nargs);
	m_vectorized.resize(nargs);

	for(unsigned i=2; i<nargs; ++i) {
		m_objects[i] = static_cast<boost::python::object>(py_args[i]).ptr();
		m_vectorized[i] = checkInputVector(m_objects[i], vlen);
	}

	/* Get the neuron index, if it's a scalar */
	unsigned neuron_index = 0;
	if(!m_vectorized[2]) { 
		neuron_index = extract<unsigned>(py_args[2]);
	}

	/* Get all scalar parameters and state variables */
	m_args.resize(nargs);
	for(unsigned i=3; i<nargs; ++i) {
		if(!m_vectorized[i]) {
			m_args[i] = extract<float>(m_objects[i]);
		}
	}

	if(vlen == 0) {
		/* All inputs are scalars, the 'scalars' array has already been
		 * populated. */
		//! \todo deal with empty list 
		nemo::Network::addNeuron(neuron_type, neuron_index, nargs-3, &m_args[3]);
	} else {
		/* At least some inputs are vectors */
		for(unsigned i=0; i < vlen; ++i) {
			/* Fill in the vector arguments */
			if(m_vectorized[2]) { 
				neuron_index = extract<unsigned>(PyList_GetItem(m_objects[2], i));
			}
			for(unsigned j=3; j<nargs; ++j) {
				if(m_vectorized[j]) {
					m_args[j] = extract<float>(PyList_GetItem(m_objects[j], i));
				}
			}
			nemo::Network::addNeuron(neuron_type, neuron_index, nargs-3, &m_args[3]);
		}
	}
	return object();
}


	} // end namespace python
} // end namespace nemo
