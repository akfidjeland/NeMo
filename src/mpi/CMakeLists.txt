# check for MPI library

FIND_PACKAGE(MPI)

IF(NOT MPI_FOUND)
	MESSAGE(SEND_ERROR "MPI enabled, but no MPI implementation found on this system")
ENDIF(NOT MPI_FOUND)

INCLUDE_DIRECTORIES(
	${MPI_INCLUDE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/src/examples
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src # for configuration
	${Boost_INCLUDE_DIR}
)

ADD_LIBRARY(nemo_mpi SHARED
	Master.cpp
	Worker.cpp
	Mapper.cpp
	SpikeQueue.cpp
	# TODO: only include if mpi timing is enabled
	MpiTimer.cpp
)

SET_TARGET_PROPERTIES(nemo_mpi PROPERTIES
		DEFINE_SYMBOL NEMO_EXPORTS
)


# TODO: should probably link against boost libraries here.
TARGET_LINK_LIBRARIES(nemo_mpi nemo ${MPI_LIBRARIES})

SET(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
ADD_EXECUTABLE(mpi_example
	example.cpp
	${EXAMPLES_DIR}/random.cpp
)

# TODO: extract only mpi and serialisation here.
TARGET_LINK_LIBRARIES(mpi_example nemo_mpi nemo ${MPI_LIBRARIES} ${Boost_LIBRARIES})

# TODO: can share this code with examples cmakefile
IF(UNIX)
	SET(INSTALL_EXAMPLE_BIN_DIR "share/nemo/examples")
ENDIF(UNIX)

IF(WIN32)
	SET(INSTALL_EXAMPLE_BIN_DIR "examples")
ENDIF(WIN32)

INSTALL(TARGETS nemo_mpi DESTINATION ${INSTALL_LIB_DIR})
INSTALL(TARGETS mpi_example DESTINATION ${INSTALL_EXAMPLE_BIN_DIR})
