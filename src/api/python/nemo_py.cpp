#include <sstream>
#include <stdexcept>
#include <functional>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <nemo.hpp>

#include "docstrings.h" // auto-generated

using namespace boost::python;


/* Py_ssize_t only introduced in Python 2.5 */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#	define PY_SSIZE_T_MAX INT_MAX
#	define PY_SSIZE_T_MIN INT_MIN
#endif


/* The simulation is only created via a factory and only accessed throught the
 * returned pointer */
boost::shared_ptr<nemo::Simulation>
makeSimulation(const nemo::Network& net, const nemo::Configuration& conf)
{
	return boost::shared_ptr<nemo::Simulation>(simulation(net, conf));
}



template<typename T>
std::string
std_vector_str(std::vector<T>& self)
{
	std::stringstream out;

	if(self.size() > 0) {
		out << "[";
		if(self.size() > 1) {
			std::copy(self.begin(), self.end() - 1, std::ostream_iterator<T>(out, ", "));
		}
		out << self.back() << "]";
	}

	return out.str();
}



/* We use uchar to stand in for booleans */
std::string
std_bool_vector_str(std::vector<unsigned char>& self)
{
	std::stringstream out;

	if(self.size() > 0) {
		out << "[";
		for(std::vector<unsigned char>::const_iterator i = self.begin(); i != self.end() - 1; ++i) {
			out << (*i ? "True" : "False") << ", ";

		}
		out << (self.back() ? "True" : "False") << ", ";
	}
	return out.str();
}



/* Converter from python pair to std::pair */
template<typename T1, typename T2>
struct from_py_list_of_pairs
{
	typedef std::pair<T1, T2> pair_t;
	typedef std::vector<pair_t> vector_t;

	from_py_list_of_pairs() {
		converter::registry::push_back(
			&convertible,
			&construct,
			boost::python::type_id<vector_t>()
		);
	}

	static void* convertible(PyObject* obj_ptr) {
		if (!PyList_Check(obj_ptr)) {
			return 0;
		}
		/* It's possible for the list to contain data of different type. In
         * that case, fall over later, during the actual conversion. */
		return obj_ptr;
	}

	/* Convert obj_ptr into a std::vector */
	static void construct(
			PyObject* list,
			boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		/* grab pointer to memory into which to construct vector */
		typedef converter::rvalue_from_python_storage<vector_t> storage_t;
		void* storage = reinterpret_cast<storage_t*>(data)->storage.bytes;

		/* in-place construct vector */
		Py_ssize_t len = PyList_Size(list);
		// vector_t* instance = new (storage) vector_t(len, 0);
		vector_t* instance = new (storage) vector_t(len);

		/* stash the memory chunk pointer for later use by boost.python */
		data->convertible = storage;

		/* populate the vector */
		vector_t& vec = *instance;
		for(unsigned i=0, i_end=len; i != i_end; ++i) {
			PyObject* pair = PyList_GetItem(list, i);
			PyObject* first = PyTuple_GetItem(pair, 0);
			PyObject* second = PyTuple_GetItem(pair, 1);
			vec[i] = std::make_pair<T1, T2>(extract<T1>(first), extract<T2>(second));
		}
	}
};


/* Python list to std::vector convertor */
template<typename T>
struct from_py_list
{
	typedef std::vector<T> vector_t;

	from_py_list() {
		converter::registry::push_back(
			&convertible,
			&construct,
			boost::python::type_id<vector_t>()
		);
	}

	static void* convertible(PyObject* obj_ptr) {
		if (!PyList_Check(obj_ptr)) {
			return 0;
		}
		/* It's possible for the list to contain data of different type. In
         * that case, fall over later, during the actual conversion. */
		return obj_ptr;
	}

	/* Convert obj_ptr into a std::vector */
	static void construct(
			PyObject* list,
			boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		/* grab pointer to memory into which to construct vector */
		typedef converter::rvalue_from_python_storage<vector_t> storage_t;
		void* storage = reinterpret_cast<storage_t*>(data)->storage.bytes;

		/* in-place construct vector */
		Py_ssize_t len = PyList_Size(list);
		vector_t* instance = new (storage) vector_t(len, 0);

		/* stash the memory chunk pointer for later use by boost.python */
		data->convertible = storage;

		/* populate the vector */
		vector_t& vec = *instance;
		for(unsigned i=0, i_end=len; i != i_end; ++i) {
			vec[i] = extract<T>(PyList_GetItem(list, i));
		}
	}
};



/*!
 * Determine if input is scalar or vector. If it is a vector, verify that the
 * vector length is the same as other vectors (whose length is already set in
 * \a vectorLength.
 *
 * \param obj either a scalar or a vector
 * \param vectorLength length of any other vectors in the same parameter list,
 * 		or '0' if there are no others.
 * \return true if the object is vector (Python list), false if it's a scalar
 */
inline
bool
checkInputVector(PyObject* obj, unsigned &vectorLength)
{
	unsigned length = PyList_Check(obj) ? PyList_Size(obj) : 0;
	if(length > 0) {
		if(vectorLength > 0 && length != vectorLength) {
			throw std::invalid_argument("input vectors of different length");
		}
		vectorLength = length;
	}
	return length > 0;
}



/*! Add one or more synapses
 *
 * \return synapse id
 *
 * The arguments (other than net) may be either scalar or vector. All vectors
 * must be of the same length. If any of the inputs are vectors, the scalar
 * arguments are replicated for each synapse.
 */
PyObject*
add_synapse(nemo::Network& net, PyObject* sources, PyObject* targets,
		PyObject* delays, PyObject* weights, PyObject* plastics)
{
	unsigned len = 0;

	bool vectorSources = checkInputVector(sources, len);
	bool vectorTargets = checkInputVector(targets, len);
	bool vectorDelays = checkInputVector(delays, len);
	bool vectorWeights = checkInputVector(weights, len);
	bool vectorPlastics = checkInputVector(plastics, len);

	to_python_value<synapse_id&> get_id;

	if(len == 0) {
		/* All inputs are scalars */
		return get_id(net.addSynapse(
					extract<unsigned>(sources),
					extract<unsigned>(targets),
					extract<unsigned>(delays),
					extract<float>(weights),
					extract<unsigned char>(plastics))
				);
	} else {
		/* At least some inputs are vectors, so we need to return a list */
		PyObject* list = PyList_New(len);
		for(unsigned i=0; i != len; ++i) {
			unsigned source = extract<unsigned>(vectorSources ? PyList_GetItem(sources, i) : sources);
			unsigned target = extract<unsigned>(vectorTargets ? PyList_GetItem(targets, i) : targets);
			unsigned delay = extract<unsigned>(vectorDelays ? PyList_GetItem(delays, i) : delays);
			float weight = extract<float>(vectorWeights ? PyList_GetItem(weights, i) : weights);
			unsigned char plastic = extract<unsigned char>(vectorPlastics ? PyList_GetItem(plastics, i) : plastics);
			PyList_SetItem(list, i, get_id(net.addSynapse(source, target, delay, weight, plastic)));
		}
		return list;
	}
}



/*! Add one or more neurons
 *
 * The arguments (other than net) may be either scalar or vector. All vectors
 * must be of the same length. If any of the inputs are vectors, the scalar
 * arguments are replicated for each synapse.
 */
void
add_neuron(nemo::Network& net, PyObject* idxs,
		PyObject* as, PyObject* bs, PyObject* cs, PyObject* ds,
		PyObject* us, PyObject* vs, PyObject* ss)
{
	unsigned len = 0;

	bool vectorIdx = checkInputVector(idxs, len);
	bool vectorA = checkInputVector(as, len);
	bool vectorB = checkInputVector(bs, len);
	bool vectorC = checkInputVector(cs, len);
	bool vectorD = checkInputVector(ds, len);
	bool vectorU = checkInputVector(us, len);
	bool vectorV = checkInputVector(vs, len);
	bool vectorS = checkInputVector(ss, len);

	if(len == 0) {
		/* All inputs are scalars */
		net.addNeuron(extract<unsigned>(idxs),
					extract<float>(as), extract<float>(bs),
					extract<float>(cs), extract<float>(ds),
					extract<float>(us), extract<float>(vs),
					extract<float>(ss));
	} else {
		/* At least some inputs are vectors */
		for(unsigned i=0; i != len; ++i) {
			unsigned idx = extract<unsigned>(vectorIdx ? PyList_GetItem(idxs, i) : idxs);
			float a = extract<float>(vectorA ? PyList_GetItem(as, i) : as);
			float b = extract<float>(vectorB ? PyList_GetItem(bs, i) : bs);
			float c = extract<float>(vectorC ? PyList_GetItem(cs, i) : cs);
			float d = extract<float>(vectorD ? PyList_GetItem(ds, i) : ds);
			float u = extract<float>(vectorU ? PyList_GetItem(us, i) : us);
			float v = extract<float>(vectorV ? PyList_GetItem(vs, i) : vs);
			float s = extract<float>(vectorS ? PyList_GetItem(ss, i) : ss);
			net.addNeuron(idx, a, b, c, d, u, v, s);
		}
	}
}



/*! Modify one or more neurons
 *
 * The arguments (other than net) may be either scalar or vector. All vectors
 * must be of the same length. If any of the inputs are vectors, the scalar
 * arguments are replicated for each synapse.
 */
template<class T>
void
set_neuron(T& net, PyObject* idxs,
		PyObject* as, PyObject* bs, PyObject* cs, PyObject* ds,
		PyObject* us, PyObject* vs, PyObject* ss)
{
	unsigned len = 0;

	bool vectorIdx = checkInputVector(idxs, len);
	bool vectorA = checkInputVector(as, len);
	bool vectorB = checkInputVector(bs, len);
	bool vectorC = checkInputVector(cs, len);
	bool vectorD = checkInputVector(ds, len);
	bool vectorU = checkInputVector(us, len);
	bool vectorV = checkInputVector(vs, len);
	bool vectorS = checkInputVector(ss, len);

	if(len == 0) {
		/* All inputs are scalars */
		net.setNeuron(extract<unsigned>(idxs),
					extract<float>(as), extract<float>(bs),
					extract<float>(cs), extract<float>(ds),
					extract<float>(us), extract<float>(vs),
					extract<float>(ss));
	} else {
		/* At least some inputs are vectors */
		for(unsigned i=0; i != len; ++i) {
			unsigned idx = extract<unsigned>(vectorIdx ? PyList_GetItem(idxs, i) : idxs);
			float a = extract<float>(vectorA ? PyList_GetItem(as, i) : as);
			float b = extract<float>(vectorB ? PyList_GetItem(bs, i) : bs);
			float c = extract<float>(vectorC ? PyList_GetItem(cs, i) : cs);
			float d = extract<float>(vectorD ? PyList_GetItem(ds, i) : ds);
			float u = extract<float>(vectorU ? PyList_GetItem(us, i) : us);
			float v = extract<float>(vectorV ? PyList_GetItem(vs, i) : vs);
			float s = extract<float>(vectorS ? PyList_GetItem(ss, i) : ss);
			net.setNeuron(idx, a, b, c, d, u, v, s);
		}
	}
}





unsigned
set_neuron_x_length(PyObject* a, PyObject* b)
{
	unsigned len = 0;

	bool vectorA = checkInputVector(a, len);
	bool vectorB = checkInputVector(b, len);

	if(vectorA != vectorB) {
		throw std::invalid_argument("first and third argument must either both be scalar or lists of same length");
	}
	return len;
}



/*! Set neuron parameters for one or more neurons
 *
 * On the Python side the syntax is net.set_neuron_parameter(neurons, param,
 * values). Either these are all scalar, or neurons and values are both lists
 * of the same length.
 */
template<class T>
void
set_neuron_parameter(T& obj, PyObject* neurons, unsigned param, PyObject* values)
{
	const unsigned len = set_neuron_x_length(neurons, values);
	if(len == 0) {
		obj.setNeuronParameter(extract<unsigned>(neurons), param, extract<float>(values));
	} else {
		for(unsigned i=0; i < len; ++i) {
			unsigned neuron = extract<unsigned>(PyList_GetItem(neurons, i));
			float value = extract<float>(PyList_GetItem(values, i));
			obj.setNeuronParameter(neuron, param, value);
		}
	}
}



/*! Set neuron state for one or more neurons
 *
 * On the Python side the syntax is net.set_neuron_state(neurons, param,
 * values). Either these are all scalar, or neurons and values are both lists
 * of the same length.
 */
template<class T>
void
set_neuron_state(T& obj, PyObject* neurons, unsigned param, PyObject* values)
{
	const unsigned len = set_neuron_x_length(neurons, values);
	if(len == 0) {
		obj.setNeuronState(extract<unsigned>(neurons), param, extract<float>(values));
	} else {
		for(unsigned i=0; i < len; ++i) {
			unsigned neuron = extract<unsigned>(PyList_GetItem(neurons, i));
			float value = extract<float>(PyList_GetItem(values, i));
			obj.setNeuronState(neuron, param, value);
		}
	}
}



template<class T>
PyObject*
get_neuron_parameter(T& obj, PyObject* neurons, unsigned param)
{
	const Py_ssize_t len = PyList_Check(neurons) ? PyList_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(obj.getNeuronParameter(extract<unsigned>(neurons), param));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PyList_GetItem(neurons, i));
			const float val = obj.getNeuronParameter(neuron, param);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



template<class T>
PyObject*
get_neuron_state(T& obj, PyObject* neurons, unsigned param)
{
	const Py_ssize_t len = PyList_Check(neurons) ? PyList_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(obj.getNeuronState(extract<unsigned>(neurons), param));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PyList_GetItem(neurons, i));
			const float val = obj.getNeuronState(neuron, param);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



/*! Return the membrane potential of one or more neurons */
PyObject*
get_membrane_potential(nemo::Simulation& sim, PyObject* neurons)
{
	const Py_ssize_t len = PyList_Check(neurons) ? PyList_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(sim.getMembranePotential(extract<unsigned>(neurons)));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PyList_GetItem(neurons, i));
			const float val = sim.getMembranePotential(neuron);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



/* Convert scalar type to corresponding C++ type. Oddly, boost::python does not
 * seem to have this */
template<typename T>
PyObject*
insert(T)
{
	throw std::logic_error("invalid static type conversion in Python/C++ interface");
}


template<> PyObject* insert<float>(float x) { return PyFloat_FromDouble(x); }
template<> PyObject* insert<unsigned>(unsigned x) { return PyInt_FromLong(x); }
template<> PyObject* insert<unsigned char>(unsigned char x) { return PyBool_FromLong(x); }



/*! Return scalar or vector synapse parameter/state of type R from a
 * ReadableNetwork instance of type Net */
template<typename T>
PyObject*
get_synapse_x(const nemo::ReadableNetwork& net,
		PyObject* ids,
		std::const_mem_fun1_ref_t<T, nemo::ReadableNetwork, const synapse_id&> get_x)
{
	const Py_ssize_t len = PySequence_Check(ids) ? PySequence_Size(ids) : 0;
	if(len == 0) {
		return insert<T>(get_x(net, extract<synapse_id>(ids)));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			synapse_id id = extract<synapse_id>(PySequence_GetItem(ids, i));
			const T val = get_x(net, id);
			PyList_SetItem(list, i, insert<T>(val));
		}
		return list;
	}
}



template<class Net>
PyObject*
get_synapse_source(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseSource));
}


template<class Net>
PyObject*
get_synapse_target(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseTarget));
}


template<class Net>
PyObject*
get_synapse_delay(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseDelay));
}


template<class Net>
PyObject*
get_synapse_weight(const Net& net, PyObject* ids)
{
	return get_synapse_x<float>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseWeight));
}


template<class Net>
PyObject*
get_synapse_plastic(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned char>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapsePlastic));
}



/* This wrappers for overloads of nemo::Simulation::step */
const std::vector<unsigned>&
step_noinput(nemo::Simulation& sim)
{
	return sim.step();
}


const std::vector<unsigned>&
step_f(nemo::Simulation& sim, const std::vector<unsigned>& fstim)
{
	return sim.step(fstim);
}


const std::vector<unsigned>&
step_i(nemo::Simulation& sim, const std::vector< std::pair<unsigned, float> >& istim)
{
	return sim.step(istim);
}


const std::vector<unsigned>&
step_fi(nemo::Simulation& sim,
		const std::vector<unsigned>& fstim,
		const std::vector< std::pair<unsigned, float> >& istim)
{
	return sim.step(fstim, istim);
}


void
initializeConverters()
{
	// register the from-python converter
	from_py_list<synapse_id>();
	from_py_list<unsigned>();
	from_py_list<unsigned char>();
	from_py_list<float>();
	from_py_list_of_pairs<unsigned, float>();
}


/* The STDP configuration comes in two forms in the C++ API. Use just the
 * original form here, in order to avoid breaking existing code. */
void (nemo::Configuration::*stdp2)(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight) = &nemo::Configuration::setStdpFunction;


BOOST_PYTHON_MODULE(_nemo)
{
	def("init", initializeConverters);

	class_<std::vector<unsigned> >("std_vector_unsigned")
		.def(vector_indexing_suite<std::vector<unsigned> >())
		.def("__str__", &std_vector_str<unsigned>)
	;

	class_<std::vector<float> >("std_vector_float")
		.def(vector_indexing_suite<std::vector<float> >())
		.def("__str__", &std_vector_str<float>)
	;

	class_<std::vector<unsigned char> >("std_vector_uchar")
		.def(vector_indexing_suite<std::vector<unsigned char> >())
		.def("__str__", &std_bool_vector_str)
	;

	class_<std::vector<uint64_t> >("std_vector_uint64")
		.def(vector_indexing_suite<std::vector<uint64_t> >())
		.def("__str__", &std_vector_str<uint64_t>)
	;

	class_<nemo::Configuration>("Configuration", CONFIGURATION_DOC)
		//.def("enable_logging", &nemo::Configuration::enableLogging)
		//.def("disable_logging", &nemo::Configuration::disableLogging)
		//.def("logging_enabled", &nemo::Configuration::loggingEnabled)
		.def("set_stdp_function", stdp2, CONFIGURATION_SET_STDP_FUNCTION_DOC)
		.def("set_cuda_backend", &nemo::Configuration::setCudaBackend, CONFIGURATION_SET_CUDA_BACKEND_DOC)
		.def("set_cpu_backend", &nemo::Configuration::setCpuBackend, CONFIGURATION_SET_CPU_BACKEND_DOC)
		.def("backend_description", &nemo::Configuration::backendDescription, CONFIGURATION_BACKEND_DESCRIPTION_DOC)
	;

	class_<nemo::Network, boost::noncopyable>("Network", NETWORK_DOC)
		.def("add_neuron", add_neuron, NETWORK_ADD_NEURON_DOC)
		.def("add_synapse", add_synapse, NETWORK_ADD_SYNAPSE_DOC)
		.def("set_neuron", set_neuron<nemo::Network>, CONSTRUCTABLE_SET_NEURON_DOC)
		.def("get_neuron_state", get_neuron_state<nemo::Network>, CONSTRUCTABLE_GET_NEURON_STATE_DOC)
		.def("get_neuron_parameter", get_neuron_parameter<nemo::Network>, CONSTRUCTABLE_GET_NEURON_PARAMETER_DOC)
		.def("set_neuron_state", set_neuron_state<nemo::Network>, CONSTRUCTABLE_SET_NEURON_STATE_DOC)
		.def("set_neuron_parameter", set_neuron_parameter<nemo::Network>, CONSTRUCTABLE_SET_NEURON_PARAMETER_DOC)
		.def("get_synapse_source", &nemo::Network::getSynapseSource)
		.def("neuron_count", &nemo::Network::neuronCount, NETWORK_NEURON_COUNT_DOC)
		.def("get_synapses_from", &nemo::Network::getSynapsesFrom, return_value_policy<copy_const_reference>(), CONSTRUCTABLE_GET_SYNAPSES_FROM_DOC)
		.def("get_synapse_source", get_synapse_source<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_SOURCE_DOC)
		.def("get_synapse_target", get_synapse_target<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_TARGET_DOC)
		.def("get_synapse_delay", get_synapse_delay<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_DELAY_DOC)
		.def("get_synapse_weight", get_synapse_weight<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_WEIGHT_DOC)
		.def("get_synapse_plastic", get_synapse_plastic<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_PLASTIC_DOC)
	;

	class_<nemo::Simulation, boost::shared_ptr<nemo::Simulation>, boost::noncopyable>(
			"Simulation", SIMULATION_DOC, no_init)
		.def("__init__", make_constructor(makeSimulation))
		/* For the step function(s) named optional input arguments is handled
		 * in pure python. See __init__.py. */
		/* May want to make a copy here, for some added safety:
		 * return_value_policy<copy_const_reference>()
		 *
		 * In the current form we return a reference to memory handled by the
		 * simulation object, which may be overwritten by subsequent calls to
		 * to this function. */
		.def("step_noinput", step_noinput, return_internal_reference<1>())
		.def("step_f", step_f, return_internal_reference<1>())
		.def("step_i", step_i, return_internal_reference<1>())
		.def("step_fi", step_fi, return_internal_reference<1>())
		.def("apply_stdp", &nemo::Simulation::applyStdp, SIMULATION_APPLY_STDP_DOC)
		.def("set_neuron", set_neuron<nemo::Simulation>, CONSTRUCTABLE_SET_NEURON_DOC)
		.def("get_neuron_state", get_neuron_state<nemo::Simulation>, CONSTRUCTABLE_GET_NEURON_STATE_DOC)
		.def("get_neuron_parameter", get_neuron_parameter<nemo::Simulation>, CONSTRUCTABLE_GET_NEURON_PARAMETER_DOC)
		.def("set_neuron_state", set_neuron_state<nemo::Simulation>, CONSTRUCTABLE_SET_NEURON_STATE_DOC)
		.def("set_neuron_parameter", set_neuron_parameter<nemo::Simulation>, CONSTRUCTABLE_SET_NEURON_PARAMETER_DOC)
		.def("get_membrane_potential", get_membrane_potential, SIMULATION_GET_MEMBRANE_POTENTIAL_DOC)
		.def("get_synapses_from", &nemo::Simulation::getSynapsesFrom, return_value_policy<copy_const_reference>(), CONSTRUCTABLE_GET_SYNAPSES_FROM_DOC)
		.def("get_synapse_source", get_synapse_source<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_SOURCE_DOC)
		.def("get_synapse_target", get_synapse_target<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_TARGET_DOC)
		.def("get_synapse_delay", get_synapse_delay<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_DELAY_DOC)
		.def("get_synapse_weight", get_synapse_weight<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_WEIGHT_DOC)
		.def("get_synapse_plastic", get_synapse_plastic<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_PLASTIC_DOC)
		.def("elapsed_wallclock", &nemo::Simulation::elapsedWallclock, SIMULATION_ELAPSED_WALLCLOCK_DOC)
		.def("elapsed_simulation", &nemo::Simulation::elapsedSimulation, SIMULATION_ELAPSED_SIMULATION_DOC)
		.def("reset_timer", &nemo::Simulation::resetTimer, SIMULATION_RESET_TIMER_DOC)
	;
}
