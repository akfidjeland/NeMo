FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
SET(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/base
	${CMAKE_BINARY_DIR}/base # for configuration files
	${Boost_INCLUDE_DIR}
	${EXAMPLES_DIR})
ADD_DEFINITIONS("-L${CMAKE_BINARY_DIR}/src")
ADD_EXECUTABLE(test
	test.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random1k.cpp)
TARGET_LINK_LIBRARIES(test nemo ${Boost_LIBRARIES})
ADD_EXECUTABLE(rtest
	rtest.cpp
	${EXAMPLES_DIR}/torus.cpp
	${EXAMPLES_DIR}/random1k.cpp)
TARGET_LINK_LIBRARIES(rtest nemo)
