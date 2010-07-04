INCLUDE_DIRECTORIES(
	${CMAKE_BINARY_DIR}/base # for configuration files
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}
)

SET(BUILD_SHARED_LIBS TRUE)

ADD_LIBRARY(nemo_base
	nemo_c.cpp
	Configuration.cpp
	ConfigurationImpl.cpp
	Network.cpp
	NetworkImpl.cpp
	Simulation.cpp
	SimulationBackend.cpp
	ConnectivityMatrix.cpp
	fixedpoint.cpp
)

SET_TARGET_PROPERTIES(nemo_base PROPERTIES DEFINE_SYMBOL NEMO_EXPORTS)


ADD_LIBRARY(nemo nemo.cpp)
SET_TARGET_PROPERTIES(nemo PROPERTIES DEFINE_SYMBOL NEMO_EXPORTS)
TARGET_LINK_LIBRARIES(nemo nemo_base nemo_cuda)


INSTALL(TARGETS nemo nemo_base DESTINATION ${INSTALL_LIB_DIR})

INSTALL(FILES 
		nemo.h
		nemo.hpp
		nemo_error.h
		Configuration.hpp
		Network.hpp
		Simulation.hpp
		${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h
	DESTINATION ${INSTALL_INCLUDE_DIR}
)


##############################################################################
# General system configuration

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckFunctionExists)

# CheckFunctionExists does not seem to work with GCC builtins
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clzll(0); }" HAVE_BUILTIN_CLZLL)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_clz(0); }" HAVE_BUILTIN_CLZ)
CHECK_CXX_SOURCE_COMPILES("int main() { __builtin_ctzll(0); }" HAVE_BUILTIN_CTZLL)

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
	${CMAKE_CURRENT_SOURCE_DIR}/nemo_config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h)
