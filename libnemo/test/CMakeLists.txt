FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
SET(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}
	${CMAKE_BINARY_DIR} # for configuration files
	${EXAMPLES_DIR}
	${Boost_INCLUDE_DIR}
)

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF(NOT Boost_USE_STATIC_LIBS)
	ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
ENDIF(NOT Boost_USE_STATIC_LIBS)

LINK_STATIC_MSVCR()

ADD_EXECUTABLE(test
	test.cpp
	stdp.cpp
	utils.cpp
	networks.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random.cpp)
TARGET_LINK_LIBRARIES(test nemo ${Boost_LIBRARIES})

ADD_EXECUTABLE(rtest
	rtest.cpp
	utils.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random.cpp)
TARGET_LINK_LIBRARIES(rtest nemo ${Boost_LIBRARIES})

IF(INCLUDE_MPI)
	ADD_EXECUTABLE(stest serialisation.cpp)
	TARGET_LINK_LIBRARIES(stest nemo ${Boost_LIBRARIES})
ENDIF(INCLUDE_MPI)

ADD_EXECUTABLE(api_test test_c_api.cpp utils.cpp)
TARGET_LINK_LIBRARIES(api_test nemo ${Boost_LIBRARIES})

