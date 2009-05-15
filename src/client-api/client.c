#include "client.h"
#include <stdio.h>

#ifdef __GLASGOW_HASKELL__
extern void __stginit_Client(void);
#endif


void
Client_init(void)
{
	static int argc = 1;
	static char* argv[] = { "nemoclient", NULL }, **argv_ = argv;
    hs_init(&argc, &argv_);
#ifdef __GLASGOW_HASKELL__
	hs_add_root(__stginit_Client);
#endif
}


void
Client_exit(void)
{
	hs_exit();
}
