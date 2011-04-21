/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include <boost/format.hpp>

#include <nemo/config.h>
#include "Plugin.hpp"
#include "exception.hpp"

#include "plugin.ipp"

#ifdef _MSC_VER
const char DIRSEP_CHAR = '\\';
const char* HOME_ENV_VAR = "userprofile";
#else
const char DIRSEP_CHAR = '/';
const char* HOME_ENV_VAR = "HOME";
#endif

namespace nemo {


Plugin::Plugin(const std::string& name) :
	m_handle(NULL)
{
	init(name);
	try {
		load(name);
	} catch(nemo::exception& e) {
		dl_exit();
		throw;
	}
}



Plugin::Plugin(const std::string& name, const std::string& subdir) :
	m_handle(NULL)
{
	init(name);
	try {
		setpath(subdir);
		load(name);
	} catch(nemo::exception& e) {
		dl_exit();
		throw;
	}
}



Plugin::~Plugin()
{
	/* Both the 'unload' and the 'exit' can fail. There's not much we can do
	 * about either situation, so just continue on our merry way */
	dl_unload(m_handle);
	dl_exit();
}




void
Plugin::init(const std::string& name)
{
	using boost::format;
	if(!dl_init()) {
		/* Nothing to clean up in case of failure here */
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



void
Plugin::setpath(const std::string& subdir)
{
	using boost::format;

	char* home = getenv(HOME_ENV_VAR);
	if(home == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				"Could not locate user's home directory when searching for plugins");
	}

	const std::string userPath = str(format("%s%c%s%c%s")
			% home % DIRSEP_CHAR
			% NEMO_USER_PLUGIN_DIR % DIRSEP_CHAR
			% subdir);
	if(!dl_setsearchpath(userPath.c_str())) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when setting user plugin search path (%s): %s")
					% userPath % dl_error()));
	}

	const std::string systemPath = str(format("%s%c%s")
			% NEMO_SYSTEM_PLUGIN_DIR % DIRSEP_CHAR % subdir);
	if(!dl_addsearchdir(systemPath.c_str())) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when setting system plugin search path (%s): %s")
					% userPath % dl_error()));
	}
}



void
Plugin::load(const std::string& name)
{
	using boost::format;
	m_handle = dl_load(dl_libname(name).c_str());
	if(m_handle == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



void*
Plugin::function(const std::string& name) const
{
	void* fn = dl_sym(m_handle, name.c_str());
	if(fn == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	return fn;
}


}
