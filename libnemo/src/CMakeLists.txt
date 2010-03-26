# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.
##############################################################################

# Boost libraries
SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREAD OFF)
FIND_PACKAGE(Boost REQUIRED)

##############################################################################
# CUDA configuration

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
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

CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# C++ only includes
INCLUDE_DIRECTORIES(
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
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



##############################################################################
SET(BUILD_SHARED_LIBS TRUE)

CUDA_ADD_LIBRARY(nemo
	kernel.cu
	nemo.cpp
	ConnectivityMatrix.cpp
	CudaNetwork.cpp
	CycleCounters.cpp
	DeviceAssertions.cpp
	FiringOutput.cpp
	fixedpoint.cpp
	Incoming.cpp
	NeuronParameters.cpp
	Network.cpp
	Outgoing.cpp
	RSMatrix.cpp
	SynapseAddressTable.cpp
	ThalamicInput.cpp
	WarpAddressTable.cpp
	time.cpp
  )

#SET_TARGET_PROPERTIES(nemo PROPERTIES SOVERSION 1)

# A number of .cu files are #included in the main .cu file
ADD_DEPENDENCIES(nemo
	src/applySTDP.cu
	src/bitvector.cu
	src/connectivityMatrix.cu
	src/cycleCounting.cu
	src/cycle.cu
	src/debug.cu
	src/device_assert.cu
	src/double_buffer.cu
	src/fixedpoint.cu
	src/incoming.cu
	src/outgoing.cu
	src/partitionConfiguration.cu
	src/stdp.cu
	src/step.cu
	src/thalamicInput.cu
  )


# Generating PTX files. 
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

INSTALL(TARGETS nemo DESTINATION lib)
INSTALL(FILES nemo.h nemo.hpp DESTINATION include)
