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

# FindCUDA sets a number of variables which or not of relevance to our build.
# Mark these as advanced, so as not to clutter the configuration.
MARK_AS_ADVANCED(FORCE
	FOUND_CUT
	FOUND_CUT_INCLUDE
	CUDA_SDK_INSTALL_PREFIX
	CUDA_ATTACH_VS_BUILD_RULE_TO_C
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
	libnemo.cpp
	ConnectivityMatrix.cpp
	CycleCounters.cpp
	DeviceAssertions.cpp
	FiringOutput.cpp
	fixedpoint.cpp
	Incoming.cpp
	NeuronParameters.cpp
	Network.cpp
	Outgoing.cpp
	RSMatrix.cpp
	RuntimeData.cpp
	SynapseGroup.cpp
	ThalamicInput.cpp
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
INSTALL(FILES libnemo.h DESTINATION include)



##############################################################################
# API Documentation
##############################################################################

# Users shouldn't normally need to build to documentation themselves.
OPTION(REBUILD_API_DOCS "Rebuild API documentation" OFF)

SET(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doc)
SET(DOXYGEN_MAN_OUTPUT_DIR ${DOXYGEN_OUTPUT_DIR}/man/man3)
SET(DOXYGEN_MAN_OUTPUTS
	${DOXYGEN_MAN_OUTPUT_DIR}/libnemo.h.3
	${DOXYGEN_MAN_OUTPUT_DIR}/nemo_Network.3
)

# Some extra pages, which should be cleaned but not installed
# TODO: add these to source package ignore files
SET(DOXYGEN_MAN_EXTRA_OUTPUTS
	${DOXYGEN_MAN_OUTPUT_DIR}/Network.hpp.3
)

SET(DOXYGEN_OUTPUTS
	${DOXYGEN_MAN_OUTPUTS}
	${DOXYGEN_MAN_EXTRA_OUTPUTS}
)

IF(REBUILD_API_DOCS)
	FIND_PACKAGE(Doxygen)

	IF(DOXYGEN_FOUND)

		SET(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
		CONFIGURE_FILE(${DOXYFILE_IN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ESCAPE_QUOTES @ONLY)

		# Run Doxygen
		# TODO: don't copy the whole doc-tree into the source tree. Most of
		# the generated files are not needed for the source installation.
		ADD_CUSTOM_COMMAND(
			OUTPUT ${DOXYGEN_OUTPUTS}
			COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYGEN_MAN_OUTPUT_DIR} ${CMAKE_SOURCE_DIR}/doc/man/man3
			DEPENDS
				${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
				${CMAKE_SOURCE_DIR}/src/libnemo.h
				${CMAKE_SOURCE_DIR}/src/Network.hpp
		)

		# TODO: generate PDF manual

	ENDIF(DOXYGEN_FOUND)

ELSE(REBUILD_API_DOCS)

	# The generated documentation should already be present in the source tree.
	# Copy it into the build tree.
	ADD_CUSTOM_COMMAND(
		OUTPUT ${DOXYGEN_OUTPUTS}
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc ${CMAKE_BINARY_DIR}/doc
	)

ENDIF(REBUILD_API_DOCS)


# Create all documentation
ADD_CUSTOM_TARGET(api-doc
	DEPENDS ${DOXYGEN_OUTPUTS} ${DOXYGEN_SRC_OUTPUT}
	COMMENT "Generating documentation"
	VERBATIM
)


# Always create API documentation
ADD_DEPENDENCIES(nemo api-doc)


IF(UNIX)
	INSTALL(FILES ${DOXYGEN_MAN_OUTPUTS} DESTINATION man/man3)
ENDIF(UNIX)
