INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/src # for configuration files
	${Boost_INCLUDE_DIR})
ADD_DEFINITIONS("-L${CMAKE_BINARY_DIR}/src" -DUSING_MAIN)
ADD_EXECUTABLE(torus torus.cpp sim_runner.cpp)
TARGET_LINK_LIBRARIES(torus nemo)

ADD_EXECUTABLE(random1k random1k.cpp sim_runner.cpp)
TARGET_LINK_LIBRARIES(random1k nemo)

SET(INSTALL_FILES torus.cpp sim_runner.cpp)

IF(UNIX)
	INSTALL(FILES ${INSTALL_FILES} DESTINATION share/nemo/examples)
ENDIF(UNIX)

IF(WIN32)
	INSTALL(FILES ${INSTALL_FILES} DESTINATION examples)
ENDIF(WIN32)
