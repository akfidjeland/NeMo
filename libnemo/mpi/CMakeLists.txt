# check for MPI library

FIND_PACKAGE(MPI)

IF(MPI_FOUND)

	INCLUDE_DIRECTORIES(
		${MPI_INCLUDE_PATH}
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/examples
		${CMAKE_SOURCE_DIR}/base
		${CMAKE_BINARY_DIR}/base # for configuration
		${Boost_INCLUDE_DIR}	
	)

	ADD_LIBRARY(nemo_mpi SHARED 
		Master.cpp
		Worker.cpp
		Mapper.cpp
	)

	SET_TARGET_PROPERTIES(nemo_mpi PROPERTIES
			DEFINE_SYMBOL NEMO_EXPORTS
	)

	SET(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
	ADD_EXECUTABLE(example 
		example.cpp
		${EXAMPLES_DIR}/random1k.cpp
	)

	# TODO: extract only mpi and serialisation here.
	TARGET_LINK_LIBRARIES(example ${MPI_LIBRARIES} ${Boost_LIBRARIES} nemo_mpi nemo)

	# TODO: add install targets

ELSE(MPI_FOUND)
	MESSAGE(WARNING "MPI enabled, but no MPI implementation found on this system")
ENDIF(MPI_FOUND)
