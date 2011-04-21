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

#include <string>
#include <boost/utility.hpp>

//! \todo fold functionality from dyn_load into this class
#include "dyn_load.hpp"

namespace nemo {


/* Wrapper for a dynamically loaded library or plugin */
class Plugin : private boost::noncopyable
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
