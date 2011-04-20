#ifndef NEMO_PLUGIN_HPP
#define NEMO_PLUGIN_HPP

#include <string>
#include <boost/utility.hpp>

//! \todo fold functionality from dyn_load into this class
#include "dyn_load.hpp"

namespace nemo {


/* Wrapper for a dynamically loaded library */
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
};

}

#endif
