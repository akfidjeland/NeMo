#include <boost/format.hpp>

#include "Plugin.hpp"
#include "exception.hpp"

namespace nemo {


Plugin::Plugin(const std::string& name) :
	m_handle(NULL)
{
	using boost::format;

	if(!dl_init()) {
		/* Nothing to clean up in case of failure here */
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}

	m_handle = dl_load(dl_libname(name).c_str());
	if(m_handle == NULL) {
		throw nemo::exception(NEMO_DL_ERROR,
				str(format("Error when loading plugin %s: %s")
					% dl_libname(name) % dl_error()));
	}
}



Plugin::~Plugin()
{
	/* Both the 'unload' and the 'exit' can fail. There's not much we can do
	 * about either situation, so just continue on our merry way */
	dl_unload(m_handle);
	dl_exit();
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
