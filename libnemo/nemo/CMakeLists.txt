INCLUDE(CheckCSourceCompiles)
FILE(READ "cpu/thread_affinity.c" THREAD_AFFINITY_C)
CHECK_C_SOURCE_COMPILES(${THREAD_AFFINITY_C} CAN_SET_THREAD_AFFINITY)

SET(BACKEND_DIRS "cpu")
SET(BACKEND_LIBS "nemo_cpu")
IF(NEMO_CUDA_ENABLED)
	LIST(APPEND BACKEND_DIRS "cuda")
	LIST(APPEND BACKEND_LIBS "nemo_cuda")
ENDIF(NEMO_CUDA_ENABLED)

SUBDIRS(${BACKEND_DIRS})

INCLUDE_DIRECTORIES(
	${CMAKE_BINARY_DIR}  # for configuration files
	${CMAKE_SOURCE_DIR}
	${Boost_INCLUDE_DIR}
)

SET(BUILD_SHARED_LIBS TRUE)

LINK_STATIC_MSVCR()

ADD_LIBRARY(nemo_base
	ConfigurationImpl.cpp
	Network.cpp
	NetworkImpl.cpp
	Simulation.cpp
	SimulationBackend.cpp
	ConnectivityMatrix.cpp
	RNG.cpp
	fixedpoint.cpp
)

SET_TARGET_PROPERTIES(nemo_base PROPERTIES DEFINE_SYMBOL NEMO_BASE_EXPORTS)


ADD_LIBRARY(nemo
	nemo.cpp
	nemo_c.cpp
	Configuration.cpp
)
SET_TARGET_PROPERTIES(nemo PROPERTIES DEFINE_SYMBOL NEMO_EXPORTS)
TARGET_LINK_LIBRARIES(nemo nemo_base ${BACKEND_LIBS})


INSTALL(TARGETS nemo nemo_base DESTINATION ${INSTALL_LIB_DIR})

INSTALL(FILES
		constants.h
		Configuration.hpp
		Network.hpp
		Simulation.hpp
		${CMAKE_CURRENT_BINARY_DIR}/config.h
	DESTINATION ${INSTALL_INCLUDE_DIR}/nemo
)


##############################################################################
# General system configuration

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckFunctionExists)

# CheckFunctionExists does not seem to work with GCC builtins
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clzll(0); }" HAVE_BUILTIN_CLZLL)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clz(0); }" HAVE_BUILTIN_CLZ)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_ctzll(0); }" HAVE_BUILTIN_CTZLL)
CHECK_FUNCTION_EXISTS(posix_memalign HAVE_POSIX_MEMALIGN)

# log2 is part of C99 standard, so Microsoft support is patchy
CHECK_CXX_SOURCE_COMPILES("#include <math.h>\nint main() { log2(double(0)); }" HAVE_LOG2)

IF(INCLUDE_TIMING_API)
	IF(NOT Boost_LIBRARY_DIRS)
		MESSAGE(WARNING "Boost_LIBRARY_DIRS is not set, which may cause linker errors. Either set this variable or set BOOST_ROOT")
	ELSE(NOT Boost_LIBRARY_DIRS)
		LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
	ENDIF(NOT Boost_LIBRARY_DIRS)
ENDIF(INCLUDE_TIMING_API)


# We rely on 32-bit floats
INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE(float FLOAT_SIZE)
IF(NOT ${FLOAT_SIZE} EQUAL 4)
	MESSAGE(SEND_ERROR "float does not have expected size (4B)")
ENDIF(NOT ${FLOAT_SIZE} EQUAL 4)


# Unless we're on a windows/MSVC platform we need stdint.h
IF(NOT HAVE_STDINT_H AND NOT MSVC)
	MESSAGE(FATAL_ERROR "No valid stdint.h file found on this platform")
ENDIF(NOT HAVE_STDINT_H AND NOT MSVC)


CONFIGURE_FILE(
	${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/config.h)
