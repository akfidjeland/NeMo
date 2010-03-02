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

AC_SUBST(CUDA_ENABLED_HS)
AC_SUBST(CUDA_ENABLED)


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
# Check for pthreads
########################

m4_include([m4/ax_pthread.m4])
PTHREADS_ENABLED=false
PTHREADS_ENABLED_HS="--" # i.e. comment out pthread-related parts of buildinfo file
ACX_PTHREAD([PTHREADS_ENABLED=true])

if test $PTHREADS_ENABLED = "true"; then
    PTHREADS_ENABLED_HS=
fi

AC_SUBST(PTHREADS_ENABLED_HS)


########################
# Generate output
########################


AC_CONFIG_FILES([nemo.buildinfo] [Makefile] [doc/manual/title.tex])
AC_OUTPUT
