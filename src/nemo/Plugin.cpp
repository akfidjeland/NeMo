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
const char* HOME_ENV_VAR = "userprofile";
#else
const char* HOME_ENV_VAR = "HOME";
#endif

namespace nemo {


std::vector<boost::filesystem::path> Plugin::s_extraPaths;


Plugin::Plugin(const std::string& name) :
	m_handle(NULL)
{
	init(name);
	try {
		load(name);
	} catch(nemo::exception&) {
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
	} catch(nemo::exception&) {
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



boost::filesystem::path
Plugin::userDirectory()
{
	char* home = getenv(HOME_ENV_VAR);
	if(home == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				"Could not locate user's home directory when searching for plugins");
	}

	return boost::filesystem::path(home) / NEMO_USER_PLUGIN_DIR;
}



boost::filesystem::path
Plugin::systemDirectory()
{
	using boost::format;
	using namespace boost::filesystem;

#if defined _WIN32 || defined __CYGWIN__
	/* On Windows, where there aren't any standard library locations, NeMo
	 * might be relocated. To support this, look for a plugin directory relative
	 * to the library location path rather than relative to the hard-coded
	 * installation prefix.  */
	HMODULE dll = GetModuleHandle("nemo_base.dll");
	TCHAR dllPath[MAX_PATH];
	GetModuleFileName(dll, dllPath, MAX_PATH);
	path systemPath = path(dllPath).parent_path().parent_path() / NEMO_SYSTEM_PLUGIN_DIR;
#else
	path systemPath(NEMO_SYSTEM_PLUGIN_DIR);
#endif
	if(!exists(systemPath)) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("System plugin path does not exist: %s") % systemPath));
	}
	return systemPath;
}



void
Plugin::setpath(const std::string& subdir)
{
	using boost::format;
	using namespace boost::filesystem;

	std::vector<path> paths = s_extraPaths;

	path userPath = userDirectory() / subdir;
	if(exists(userPath)) {
		paths.push_back(userPath);
	}

	path systemPath = systemDirectory() / subdir;
	paths.push_back(systemPath);

	/*! \todo there's a potential issue here when loading on Windows. Ideally
	 * we'd like to set the /exclusive/ search path for library loading here,
	 * since the same plugin has the same name for different backends. The
	 * windows API, however, seems to only ever add to the existing path.
	 *
	 * \see dl_setsearchpath
	 */
	for(std::vector<path>::const_iterator i = paths.begin();
			i != paths.end(); ++i) {
		bool success = false;
		if(i == paths.begin()) {
			success = dl_setsearchpath(i->string().c_str());
		} else {
			success = dl_addsearchdir(i->string().c_str());
		}
		if(!success) {
			throw nemo::exception(NEMO_DL_ERROR,
					str(format("Error when setting plugin search path (%s): %s")
						% (*i) % dl_error()));
		}
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



void
Plugin::addPath(const std::string& dir)
{
	using boost::format;

	boost::filesystem::path path(dir);
	if(!exists(path) && !is_directory(path)) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("User-specified path %s could not be found") % dir));
	}
	s_extraPaths.push_back(path);
}


}
