#ifndef NEMO_PLUGIN_HPP
#define NEMO_PLUGIN_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef _MSC_VER
/* Suppress generation of MSVC-specific min/max macros which otherwise break
 * std::min and std::max */
#	define NOMINMAX
#	include <windows.h>
typedef HMODULE dl_handle;
#else
#	include <ltdl.h>
typedef lt_dlhandle dl_handle;
#endif
#include <string>
#include <boost/utility.hpp>
#include <boost/filesystem.hpp>
#include <nemo/config.h>


namespace nemo {


/* Wrapper for a dynamically loaded library or plugin */
class NEMO_BASE_DLL_PUBLIC Plugin : private boost::noncopyable
{
	public :

		/*! Load a plugin from the default library path.
		 *
		 * \param name
		 * 		base name of the library, i.e. without any system-specific
		 * 		prefix or file extension. For example the library libfoo.so on
		 * 		a UNIX system has the  base name 'foo'. 
		 *
		 * \throws nemo::exception for load errors
		 */
		explicit Plugin(const std::string& name);

		/*! Load a plugin from a subdirectory of the set of NeMo-specific plugin directories
		 *
		 * Plugins are always located in a subdirectory, as they are backend-specific.
		 * There is one system plugin directory and one per-user system directory.
		 */
		Plugin(const std::string& name, const std::string& subdir);

		~Plugin();

		/*! \return function pointer for a named function
		 *
		 * The user needs to cast this to the appropriate type.
		 *
		 * \throws nemo::exception for load errors
		 */
		void* function(const std::string& name) const;

		/*! \return path to user plugin directory
		 *
		 * The path may not exist
		 */
		static boost::filesystem::path userDirectory();

		/*! \return path to system plugin directory
		 *
		 * \throws if the directory does not exist
		 */
		static boost::filesystem::path systemDirectory();

	private:

		dl_handle m_handle;

		/*! Initialise the loader */
		void init(const std::string& name);

		/*! Set NeMo-specific search paths */
		void setpath(const std::string& subdir);

		/*! Load the library */
		void load(const std::string& name);
};

}

#endif