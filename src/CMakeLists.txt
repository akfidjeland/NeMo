OPTION(NEMO_CUDA_ENABLED "Enabled simulation on CUDA devices" ON)

IF(NEMO_CUDA_ENABLED)
	OPTION(NEMO_CUDA_DYNAMIC_LOADING "Load the CUDA backend dynamically" ON)
	SET(NEMO_CUDA_DEBUG_TRACE "" CACHE STRING "Mask indicating what events to show in trace log. Or the following: 0x2 (neuron), 0x4 (synapse) 0x8 (stdp)")
	OPTION(NEMO_CUDA_DEVICE_ASSERTIONS "Enable assertions in device code" OFF)
ELSE(NEMO_CUDA_ENABLED)
	SET(NEMO_CUDA_DYNAMIC_LOADING OFF)
ENDIF(NEMO_CUDA_ENABLED)

LIST(APPEND BOOST_COMPONENTS filesystem program_options date_time)

OPTION(NEMO_MPI_ENABLED "Create MPI-enabled simulation" OFF)
IF(NEMO_MPI_ENABLED)
	LIST(APPEND BOOST_COMPONENTS mpi serialization)
	LIST(APPEND DIRS "mpi")
	OPTION(NEMO_MPI_DEBUG_TRACE "Enable debug trace for MPI backend" OFF)
	OPTION(NEMO_MPI_DEBUG_TIMING "Add detailed timing of MPI operations" OFF)
	OPTION(NEMO_MPI_COMMUNICATION_COUNTERS "Add packet counting and reporting" OFF)
ENDIF(NEMO_MPI_ENABLED)


OPTION(NEMO_CPU_DEBUG_TRACE "Add debugging trace in the CPU backend" OFF)


OPTION(NEMO_EXAMPLES_ENABLED "Build binaries for example networks" ON)
IF(NEMO_EXAMPLES_ENABLED)
	LIST(APPEND DIRS examples)
ENDIF(NEMO_EXAMPLES_ENABLED)


OPTION(NEMO_TESTS_ENABLED "Build testsuite" OFF)
IF(NEMO_TESTS_ENABLED)
	LIST(APPEND DIRS test)
ENDIF(NEMO_TESTS_ENABLED)

IF(APPLE)
	SET(Boost_USE_MULTITHREADED ON)
ELSE(APPLE)
	SET(Boost_USE_MULTITHREADED OFF)
ENDIF(APPLE)
SET(Boost_USE_MULTITHREAD ${Boost_USE_MULTITHREADED}) # for older versions of FindBoost

IF(WIN32 OR APPLE)
	SET(Boost_USE_STATIC_LIBS ON)
ELSE(WIN32 OR APPLE)
	SET(Boost_USE_STATIC_LIBS OFF)
ENDIF(WIN32 OR APPLE)

# Minimum boost version
#
# - date/time: 1.29.0
# - format:    1.29.0
# - filesystem: 1.30.0
# - optional:  1.30.0
# - program options: 1.32.0
# - smart ptr: 1.23.0
# - unordered: 1.36.0

FIND_PACKAGE(Boost 1.36.0 REQUIRED COMPONENTS ${BOOST_COMPONENTS})


# On windows just put everyting in one directory
SET(INSTALL_INCLUDE_DIR include)
IF(WIN32)
	SET(INSTALL_LIB_DIR bin)
ELSE(WIN32)
	SET(INSTALL_LIB_DIR lib)
ENDIF(WIN32)

IF(CMAKE_COMPILER_IS_GNUCC)
	SET(CMAKE_CXX_FLAGS "-Wall" ${CMAKE_CXX_FLAGS})
ENDIF(CMAKE_COMPILER_IS_GNUCC)

INCLUDE(SetRpath)
INCLUDE(LinkStaticMSVCR)

SUBDIRS(nemo api app ${DIRS})

INSTALL(FILES nemo.h nemo.hpp DESTINATION ${INSTALL_INCLUDE_DIR})

# Add uninstall target
CONFIGURE_FILE(
	"${CMAKE_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY
)

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
