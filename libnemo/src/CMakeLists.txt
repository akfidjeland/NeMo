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
SET(CUDA_HOST_COMPILATION_CPP TRUE)
FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_BUILD_EMULATION)
	ADD_DEFINITIONS(-D__DEVICE_EMULATION__)
ENDIF()

# TODO: use the definition that's automatically used by CMake
ADD_DEFINITIONS(-DBUILDING_DLL)


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

# Unless we're on a windows/MSVC platform we need stdint.h
IF(NOT HAVE_STDINT_H AND NOT MSVC)
	MESSAGE(FATAL_ERROR "No valid stdint.h file found on this platform")
ENDIF(NOT HAVE_STDINT_H AND NOT MSVC)





##############################################################################
SET(BUILD_SHARED_LIBS TRUE)

# Statically link against MSVC runtime
IF(MSVC)
	FOREACH(flag_var
		CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
		CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		IF(${flag_var} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		ENDIF(${flag_var} MATCHES "/MD")
	ENDFOREACH(flag_var)
ENDIF(MSVC)



CUDA_ADD_LIBRARY(nemo
	kernel.cu
	nemo.cpp
	Configuration.cpp
	ConnectivityMatrix.cpp
	CudaSimulation.cpp
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
  )

#SET_TARGET_PROPERTIES(nemo PROPERTIES SOVERSION 1)


# Generating PTX files. 
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()




# On windows just put everyting in one directory
IF(WIN32)
	SET(INSTALL_INCLUDE_DIR .)
	SET(INSTALL_LIB_DIR .)
ELSE(WIN32)
	SET(INSTALL_INCLUDE_DIR include/nemo)
	SET(INSTALL_LIB_DIR lib)
ENDIF(WIN32)


INSTALL(TARGETS nemo DESTINATION ${INSTALL_LIB_DIR})
INSTALL(FILES nemo.h
			Network.hpp
			Configuration.hpp
			Simulation.hpp
			STDP.hpp
			STDP.ipp
			nemo.hpp
			types.hpp
			types.h
			${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h
		DESTINATION ${INSTALL_INCLUDE_DIR})

CONFIGURE_FILE(
	${CMAKE_CURRENT_SOURCE_DIR}/nemo_config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h)
