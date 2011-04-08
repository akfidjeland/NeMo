#include "dyn_load.hpp"

#ifdef _MSC_VER

bool
dl_init()
{
	return true;
}

bool
dl_exit()
{
	return true;
}

dl_handle
dl_load(const char* name)
{
	return LoadLibrary(name);
}

bool
dl_unload(dl_handle h)
{
	return FreeLibrary(h) != 0;
}

const char*
dl_error()
{
	const char* str;
	FormatMessage(
		FORMAT_MESSAGE_FROM_SYSTEM
		| FORMAT_MESSAGE_ALLOCATE_BUFFER
		| FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, GetLastError(), 0, (LPTSTR) &str, 0, NULL);
	return str;
}

void*
dl_sym(dl_handle hdl, const char* name)
{
	return GetProcAddress(hdl, name);
}

bool
dl_setsearchpath(const char* dir)
{
	return SetDllDirectory(dir) != 0;
}

bool
dl_addsearchdir(const char* dir)
{
	return SetDllDirectory(dir) != 0;
}

std::string
dl_libname(std::string baseName)
{
	return baseName.append(".dll");
}

#else

bool
dl_init()
{
	return lt_dlinit() == 0;
}

bool
dl_exit()
{
	return lt_dlexit() == 0;
}

dl_handle
dl_load(const char* name)
{	
	return lt_dlopenext(name);
}

bool
dl_unload(dl_handle h)
{
	return lt_dlclose(h) == 0;
}

const char*
dl_error()
{
	return lt_dlerror();
}

void*
dl_sym(dl_handle hdl, const char* name)
{
	return lt_dlsym(hdl, name);
}

bool
dl_setsearchpath(const char* dir)
{
	return lt_dlsetsearchpath(dir) == 0;
}

bool
dl_addsearchdir(const char* dir)
{
	return lt_dladdsearchdir(dir) == 0;
}

std::string
dl_libname(std::string baseName)
{
	/* Leave libltdl to work out the extension */
	return std::string("lib").append(baseName);
}

#endif
