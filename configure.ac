AC_INIT

########################
# Version information
########################

# This is a bit of a roundabout way, as we read the version from cabal, and 
# then cabal passes the version back to ghc via the CPP.

PACKAGE_VERSION=`conf/version`

########################
# Check for CUDA
########################

CUDA_ENABLED=false
CUDA_ENABLED_HS="--" # i.e. comment out CUDA-part of buildinfo file
CUDA_PATH=/usr/local/cuda

# The user can override the path where we search for CUDA files
AC_ARG_WITH([cuda],
		[  --with-cuda=PATH        prefix where cuda is installed [default=/usr/local/cuda]])


if test "x$with_cuda" != x; then
	CUDA_PATH="$with_cuda"
fi

CPPFLAGS="-I$CUDA_PATH"/include

# A very basic check: if the cudart header is available we assume everything is ok. 
# TODO: also check the library
# TODO: also check that we can call the compiler
AC_CHECK_HEADERS([cuda_runtime.h], 
				 [CUDA_ENABLED=true],
				 [AC_MSG_WARN([Can't locate CUDA libraries, compiling without])])


if test $CUDA_ENABLED = "true"; then
    CUDA_ENABLED_HS=
fi

AC_SUBST(CUDA_ENABLED_HS)
AC_SUBST(CUDA_ENABLED)
AC_SUBST(CUDA_PATH)


AC_CONFIG_FILES([cuda.mk] [nemo.buildinfo])
AC_OUTPUT
