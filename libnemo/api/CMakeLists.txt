FIND_PROGRAM(MEX NAMES matlab-mex mex)	


# Set the file extension used for generated MEX libraries. MEX help reports the
# following extensions:
#
#	solaris         - .mexsol
#	hpux            - .mexhpux
#	glnx86          - .mexglx
#	glnxi64         - .mexi64
#	Mac OS X        - .mexmac
#	Windows         - .dll
# 
# It looks like Cmake does not distinguish between unices, so we assume Linux. 
MACRO(MEX_LIB_SUFFIX VAR)

	SET(${VAR} "unknown")

	IF(WIN)
		SET(${VAR} "dll")
	ENDIF(WIN)

	IF(UNIX)
		IF(CMAKE_SIZE_VOID_P EQUAL 8)
			SET(${VAR} "mexi64")
		ELSE()
			SET(${VAR} "mexglx")
		ENDIF(CMAKE_SIZE_VOID_P EQUAL 8)
	ENDIF(UNIX)

	IF(APPLE) # overrides UNIX
		SET(${VAR} "mexmac")
	ENDIF(APPLE)

ENDMACRO(MEX_LIB_SUFFIX)



IF(NOT MEX STREQUAL MEX-NOTFOUND)

	# The input sources to MEX is mostly auto-generated. This is not part of
	# the main build process since user builds should not need to do this.

	MEX_LIB_SUFFIX(SUFFIX)
	SET(MEX_INCLUDES
		-I${CMAKE_SOURCE_DIR}/src
		-I${CMAKE_BINARY_DIR}/src # for configured file
	)
	SET(MEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matlab/nemo_mex.${SUFFIX})
	SET(MEX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/matlab/nemo_mex.cpp)
	SET(MEX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/matlab)

	FILE(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

	ADD_CUSTOM_COMMAND(
		OUTPUT ${MEX_OUTPUT}
		COMMAND ${MEX} -g ${MEX_INCLUDES} -lnemo ${MEX_SOURCE} -L${CMAKE_BINARY_DIR}/src
		WORKING_DIRECTORY ${MEX_OUTPUT_DIR}
		DEPENDS ${MEX_SOURCE}
	)

	ADD_CUSTOM_TARGET(mex-api ALL DEPENDS ${MEX_OUTPUT})

	IF(WIN32)
		SET(INSTALL_DIR Matlab)
	ELSE(WIN32)
		SET(INSTALL_DIR share/nemo/matlab)
	ENDIF(WIN32)

	INSTALL(
		FILES
			${MEX_OUTPUT}
			matlab/nemoNetwork.m
			matlab/nemoConfiguration.m
			matlab/nemoSimulation.m
			matlab/example.m
		DESTINATION
			${INSTALL_DIR}
	)

ELSE(NOT MEX STREQUAL MEX-NOTFOUND)
	MESSAGE("MEX compiler not found. Skipping Matlab API generation") 
ENDIF(NOT MEX STREQUAL MEX-NOTFOUND)
