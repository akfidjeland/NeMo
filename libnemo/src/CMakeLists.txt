# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.
##############################################################################


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

##############################################################################
# CUDA configuration

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_BUILD_EMULATION)
	ADD_DEFINITIONS(-D__DEVICE_EMULATION__)
ENDIF()

# FindCUDA sets a number of variables which or not of relevance to our build.
# Mark these as advanced, so as not to clutter the configuration.
MARK_AS_ADVANCED(FORCE
	FOUND_CUT
	FOUND_CUT_INCLUDE
	CUDA_SDK_INSTALL_PREFIX
	CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
	CUDA_HOST_COMPILATION_CPP)

CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
  )

# C++ only includes
INCLUDE_DIRECTORIES(
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
	${Boost_INCLUDE_DIR}	
  )


# We need shared memory atomics
SET(CUDA_NVCC_FLAGS "-arch=sm_12")

# We rely on 32-bit floats
INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE(float FLOAT_SIZE)
IF(NOT ${FLOAT_SIZE} EQUAL 4)
	MESSAGE(SEND_ERROR "float does not have expected size (4B)")
ENDIF(NOT ${FLOAT_SIZE} EQUAL 4)

# Unless we're on a windows platform we need stdint.h
IF(NOT HAVE_STDINT_H AND NOT WIN32)
	MESSAGE(FATAL_ERROR "No valid stdint.h file found on this platform")
ENDIF(NOT HAVE_STDINT_H AND NOT WIN32)





##############################################################################
SET(BUILD_SHARED_LIBS TRUE)


CUDA_ADD_LIBRARY(nemo
	kernel.cu
	nemo.cpp
	Configuration.cpp
	ConnectivityMatrix.cpp
	CudaNetwork.cpp
	CycleCounters.cpp
	DeviceAssertions.cpp
	FiringOutput.cpp
	fixedpoint.cpp
	Incoming.cpp
	NeuronParameters.cpp
	Network.cpp
	Simulation.cpp
	Outgoing.cpp
	RSMatrix.cpp
	SynapseAddressTable.cpp
	ThalamicInput.cpp
	WarpAddressTable.cpp
	time.cpp
  )

#SET_TARGET_PROPERTIES(nemo PROPERTIES SOVERSION 1)



# Generating PTX files. 
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

INSTALL(TARGETS nemo DESTINATION lib)
INSTALL(FILES nemo.h
			Network.hpp
			Configuration.hpp
			Simulation.hpp
			STDP.hpp
			STDP.ipp
			nemo.hpp
			types.hpp
			types.h
			${CMAKE_CURRENT_BINARY_DIR}/config.h
		DESTINATION include/nemo)

CONFIGURE_FILE(
	${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/config.h)
