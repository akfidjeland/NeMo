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


########################
# Check for Thrift
########################

# Note: We can build without the thrift compiler as long as the protocol does
# not change, as the auto-generated protocol files are in the repo. We need the
# haskell libraries, but that is handled by cabal.

AC_ARG_VAR([THRIFT],
		   [Path of thrift compiler. ]
		   [This is only needed if changing the thrift protocol, ]
		   [and the compiler is not on the path])
AC_PATH_PROG([THRIFT], [thrift])

HAVE_THRIFT_COMPILER=false
if test -n "$THRIFT"; then
	HAVE_THRIFT_COMPILER=true
fi

AC_SUBST(HAVE_THRIFT_COMPILER)
AC_SUBST(THRIFT)



########################
# Generate output
########################


AC_CONFIG_FILES([cuda.mk] [nemo.buildinfo] [Makefile])
AC_OUTPUT
