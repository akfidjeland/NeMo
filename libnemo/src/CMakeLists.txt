##############################################################################
# CUDA configuration

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
SET(CUDA_HOST_COMPILATION_CPP TRUE)
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
	# TODO: can probably remove this
	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
  )

# C++ only includes
INCLUDE_DIRECTORIES(
	${CUDA_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/base
	${CMAKE_BINARY_DIR}/base # for configuration files
	${CMAKE_CURRENT_SOURCE_DIR}
	${Boost_INCLUDE_DIR}	
  )


# We need shared memory atomics
SET(CUDA_NVCC_FLAGS "-arch=sm_12")


##############################################################################
SET(BUILD_SHARED_LIBS TRUE)

# Statically link against MSVC runtime
# TODO: factor this out and used for other libraries as well
IF(MSVC)
	FOREACH(flag_var
		CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
		CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		IF(${flag_var} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		ENDIF(${flag_var} MATCHES "/MD")
	ENDFOREACH(flag_var)
ENDIF(MSVC)



CUDA_ADD_LIBRARY(nemo_cuda
	kernel.cu
	ConnectivityMatrix.cpp
	CudaSimulation.cpp
	CudaSimulationImpl.cpp
	CycleCounters.cpp
	DeviceAssertions.cpp
	FiringOutput.cpp
	Incoming.cpp
	NeuronParameters.cpp
	Mapper.cpp
	Outgoing.cpp
	RSMatrix.cpp
	SynapseAddressTable.cpp
	ThalamicInput.cpp
	WarpAddressTable.cpp
  )

SET_TARGET_PROPERTIES(nemo_cuda PROPERTIES DEFINE_SYMBOL NEMO_EXPORTS)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

INSTALL(TARGETS nemo_cuda DESTINATION ${INSTALL_LIB_DIR})
