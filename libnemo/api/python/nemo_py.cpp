#include <sstream>
#include <iterator>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <nemo.hpp>

#include "docstrings.h" // auto-generated

using namespace boost::python;


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




/* This wrappers for overloads of nemo::Simulation::step */
const std::vector<unsigned>&
step0(nemo::Simulation& sim)
{
	return sim.step();
}


const std::vector<unsigned>&
step1(nemo::Simulation& sim, const std::vector<unsigned>& fstim)
{
	return sim.step(fstim);
}



void
initializeConverters()
{
	// register the from-python converter
	from_py_list<unsigned>();
	from_py_list<float>();
}



BOOST_PYTHON_MODULE(nemo)
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

	class_<nemo::Configuration>("Configuration")
		//.def("enable_logging", &nemo::Configuration::enableLogging)
		//.def("disable_logging", &nemo::Configuration::disableLogging)
		//.def("logging_enabled", &nemo::Configuration::loggingEnabled)
		//.def("setCudaPartitionSize", &nemo::Configuration::setCudaPartitionSize)
		//.def("cudaPartitionSize", &nemo::Configuration::cudaPartitionSize)
		.def("set_stdp_function", &nemo::Configuration::setStdpFunction, CONFIGURATION_SET_STDP_FUNCTION_DOC)
		.def("set_cuda_backend", &nemo::Configuration::setCudaBackend, CONFIGURATION_SET_CUDA_BACKEND_DOC)
		.def("set_cpu_backend", &nemo::Configuration::setCpuBackend, CONFIGURATION_SET_CPU_BACKEND_DOC)
		.def("backend_description", &nemo::Configuration::backendDescription, CONFIGURATION_BACKEND_DESCRIPTION_DOC)
	;

	class_<nemo::Network, boost::noncopyable>("Network")
		.def("add_neuron", &nemo::Network::addNeuron, NETWORK_ADD_NEURON_DOC)
		.def("add_synapse", &nemo::Network::addSynapse, NETWORK_ADD_SYNAPSE_DOC)
		.def("neuron_count", &nemo::Network::neuronCount, NETWORK_NEURON_COUNT_DOC)
	;

	class_<nemo::Simulation, boost::shared_ptr<nemo::Simulation>, boost::noncopyable>("Simulation", no_init)
		.def("__init__", make_constructor(makeSimulation))
		.def("step", &nemo::Simulation::step, return_internal_reference<1>(), SIMULATION_STEP_DOC)
			/* May want to make a copy here, for some added safety:
			 * return_value_policy<copy_const_reference>() */
		.def("step", step1, return_internal_reference<1>(), SIMULATION_STEP_DOC)
		.def("step", step0, return_internal_reference<1>(), SIMULATION_STEP_DOC)
		.def("apply_stdp", &nemo::Simulation::applyStdp, SIMULATION_APPLY_STDP_DOC)
		//.def("get_synapses", &nemo::Simulation::getSynapses)
		.def("elapsed_wallclock", &nemo::Simulation::elapsedWallclock, SIMULATION_ELAPSED_WALLCLOCK_DOC)
		.def("elapsed_simulation", &nemo::Simulation::elapsedSimulation, SIMULATION_ELAPSED_SIMULATION_DOC)
		.def("reset_timer", &nemo::Simulation::resetTimer, SIMULATION_RESET_TIMER_DOC)
	;
}
