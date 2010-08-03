#ifndef NEMO_DYN_LOAD_HPP
#define NEMO_DYN_LOAD_HPP

/* Common interface for dynamic library loading. */

#ifdef _MSC_VER

#include <windows.h>
typedef HMODULE dl_handle;

#define LIB_NAME(base) base ".dll"

#else

#include <ltdl.h>
typedef lt_dlhandle dl_handle;

// leave ltdl to work out the extension
#define LIB_NAME(base) "lib" base

#endif

/*! Initialise loading routines, returning success */
bool dl_init();

/*! Shut down loading routines, returning success */
bool dl_exit();

/*! Load library, returning handle to library. Returns NULL in case of failure */
dl_handle dl_load(const char* name);

/*! Unload library. Return success. */
bool dl_unload(dl_handle h);

/*! Return description of last error */
const char* dl_error();

/* Return function pointer to given symbol or NULL if there's an error. */
void* dl_sym(dl_handle, const char* name);

#endif
