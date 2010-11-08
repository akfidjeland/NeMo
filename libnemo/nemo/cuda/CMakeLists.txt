##############################################################################
# CUDA configuration

SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
SET(CUDA_HOST_COMPILATION_CPP TRUE)
FIND_PACKAGE(CUDA REQUIRED)

IF(CUDA_BUILD_EMULATION)
	ADD_DEFINITIONS(-D__DEVICE_EMULATION__)
ENDIF()

IF(NEMO_CUDA_DEBUG_TRACE)
	IF(NOT CUDA_BUILD_EMULATION)
		IF(HAVE_CUPRINTF)
			CUDA_INCLUDE_DIRECTORIES(${CUPRINTF_INCLUDE_DIR})
			INCLUDE_DIRECTORIES(${CUPRINTF_INCLUDE_DIR})
		ELSE()
			MESSAGE(WARNING "Cannot provide debugging trace for CUDA backend. Either device emulation must be enabled (CUDA_BUILD_EMULATION) or cuPrintf must be avilable")
		ENDIF(HAVE_CUPRINTF)
	ENDIF(NOT CUDA_BUILD_EMULATION)
ENDIF(NEMO_CUDA_DEBUG_TRACE)




# FindCUDA sets a number of variables which or not of relevance to our build.
# Mark these as advanced, so as not to clutter the configuration.
MARK_AS_ADVANCED(FORCE
	FOUND_CUT
	FOUND_CUT_INCLUDE
	CUDA_SDK_INSTALL_PREFIX
	CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
	CUDA_HOST_COMPILATION_CPP)

CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}
	${CMAKE_BINARY_DIR} # for configuration files
	${Boost_INCLUDE_DIR}
  )

# C++ only includes
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}
	${CMAKE_BINARY_DIR} # for configuration files
	${CUDA_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
  )


# We need shared memory atomics
# TODO: we should generate separate version for later devices
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_12")


##############################################################################
SET(BUILD_SHARED_LIBS TRUE)

CUDA_ADD_LIBRARY(nemo_cuda SHARED
	kernel.cu
	ConnectivityMatrix.cpp
	create_simulation.cpp
	Simulation.cpp
	CycleCounters.cpp
	DeviceAssertions.cpp
	FiringBuffer.cpp
	Incoming.cpp
	Neurons.cpp
	Mapper.cpp
	Outgoing.cpp
	RSMatrix.cpp
	WarpAddressTable.cpp
	device_memory.cpp
	devices.cpp
	log.cpp
  )

SET_TARGET_PROPERTIES(nemo_cuda PROPERTIES DEFINE_SYMBOL NEMO_CUDA_EXPORTS)
TARGET_LINK_LIBRARIES(nemo_cuda nemo_base)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

INSTALL(TARGETS nemo_cuda DESTINATION ${INSTALL_LIB_DIR})
