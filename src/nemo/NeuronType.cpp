/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>

#include <nemo/config.h>

#include "NeuronType.hpp"
#include "exception.hpp"


namespace nemo {


NeuronType::NeuronType(const std::string& name) :
	mf_nParam(0), mf_nState(0),
	m_name(name), m_membranePotential(0),
	m_nrand(false), m_stateHistory(1)
{
	parseConfigurationFile(name);
}


/*! Return full name of .ini file for the given plugin
 *
 * Files are searched in both the user and the system plugin directories, in
 * that order, returning the first match.
 *
 * \throws nemo::exception if no plugin configuration file is found
 */
boost::filesystem::path
configurationFile(const std::string& name)
{
	using boost::format;
	using namespace boost::filesystem;

	path userPath = path(NEMO_USER_PLUGIN_DIR) / (name + ".ini");
	if(exists(userPath)) {
		return userPath;
	}

	path systemPath = path(NEMO_SYSTEM_PLUGIN_DIR) / (name + ".ini");
	if(exists(systemPath)) {
		return systemPath;
	}

	throw nemo::exception(NEMO_INVALID_INPUT,
			str(format("Could not find .ini file for plugin %s") % name));
}


template<typename T>
T
getRequired(boost::program_options::variables_map vm,
		const std::string& name,
		boost::filesystem::path file)
{
	using boost::format;

	if(vm.count(name) != 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Missing parameter %s in configuration file %s")
					% name % file));
	}
	return vm[name].as<T>();
}




void
NeuronType::parseConfigurationFile(const std::string& name)
{
	using boost::format;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;

	po::options_description desc("Allowed options");
	desc.add_options()
		/* required fields, no defaults */
		("parameters", po::value<unsigned>(), "number of neuron parameters")
		("state", po::value<unsigned>(), "number of neuron state variables")
		("membrane-potential", po::value<unsigned>(), "index of membrane potential variable")
		("nrand", po::value<bool>(), "is normal RNG required?")
		/* optional fields */
		("history", po::value<unsigned>()->default_value(1), "index of membrane potential variable")
	;

	fs::path filename = configurationFile(name);

	fs::fstream file(filename);

	try {
		po::variables_map vm;
		po::store(po::parse_config_file(file, desc), vm);
		po::notify(vm);

		mf_nParam = getRequired<unsigned>(vm, "parameters", filename);
		mf_nState = getRequired<unsigned>(vm, "state", filename);
		m_membranePotential = getRequired<unsigned>(vm, "membrane-potential", filename);
		m_nrand = getRequired<bool>(vm, "nrand", filename);
		m_stateHistory = vm["history"].as<unsigned>();
	} catch (po::error& e) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Error parsing neuron model configuration file %s: %s")
					% filename % e.what()));
	}
}



size_t
hash_value(const nemo::NeuronType& type)
{
	std::size_t seed = 0;
	boost::hash_combine(seed, type.mf_nParam);
	boost::hash_combine(seed, type.mf_nState);
	boost::hash_combine(seed, type.m_name);
	return seed;
}


size_t
NeuronType::hash_value() const
{
	static size_t h = nemo::hash_value(*this);
	return h;	
}

}
