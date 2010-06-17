
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
