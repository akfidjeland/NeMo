INCLUDE_DIRECTORIES(
	${CMAKE_BINARY_DIR}/base # for configuration files
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}
)

SET(BUILD_SHARED_LIBS TRUE)

ADD_LIBRARY(nemo
	nemo.cpp
	Configuration.cpp
	ConfigurationImpl.cpp
	Network.cpp
	NetworkImpl.cpp
	Simulation.cpp
)

SET_TARGET_PROPERTIES(nemo PROPERTIES
		DEFINE_SYMBOL NEMO_EXPORTS
)


TARGET_LINK_LIBRARIES(nemo nemo_cuda)


INSTALL(TARGETS nemo DESTINATION ${INSTALL_LIB_DIR})

INSTALL(FILES 
		nemo.h
		nemo.hpp
		Configuration.hpp
		Network.hpp
		Simulation.hpp
		${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h
	DESTINATION ${INSTALL_INCLUDE_DIR}
)

CONFIGURE_FILE(
	${CMAKE_CURRENT_SOURCE_DIR}/nemo_config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/nemo_config.h)
