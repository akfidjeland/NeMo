# TODO: move these into a separate CMake module

IF(WIN32)
	FIND_PROGRAM(MEX NAMES mex.bat)
ELSE(WIN32)
	FIND_PROGRAM(MEX NAMES matlab-mex mex)
ENDIF(WIN32)

# Set the file extension used for generated MEX libraries. MEX help reports the
# following extensions:
#
#	solaris         - .mexsol
#	hpux            - .mexhpux
#	glnx86          - .mexglx
#	glnxi64         - .mexi64
#	Mac OS X        - .mexmac
#	Windows         - .mexw32. Older version use .dll
# 
# It looks like Cmake does not distinguish between unices, so we assume Linux. 
MACRO(MEX_LIB_SUFFIX VAR)

	SET(${VAR} "unknown")

	IF(WIN32)
		# TODO: not sure how/if this works with 64-bit OS
		SET(${VAR} "mexw32")
	ENDIF(WIN32)

	IF(UNIX)
		IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
			# Note: this will fail on itanium
			SET(${VAR} "mexa64")
		ELSE()
			SET(${VAR} "mexglx")
		ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)
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
		-I${CMAKE_SOURCE_DIR}
		-I${CMAKE_BINARY_DIR} # for configured file
		-I${Boost_INCLUDE_DIR}
	)
	SET(MEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nemo_mex.${SUFFIX})
	SET(MEX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/nemo_mex.cpp)
	SET(NEMO_LIB_DIR ${CMAKE_BINARY_DIR}/nemo/${CMAKE_CFG_INTDIR})
	SET(MEX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

	FILE(MAKE_DIRECTORY ${MEX_OUTPUT_DIR})

	ADD_CUSTOM_COMMAND(
		OUTPUT ${MEX_OUTPUT}
		COMMAND ${MEX} ${MEX_INCLUDES} -L${NEMO_LIB_DIR} -lnemo ${MEX_SOURCE}
		WORKING_DIRECTORY ${MEX_OUTPUT_DIR}
		DEPENDS ${MEX_SOURCE}
	)

	ADD_CUSTOM_TARGET(mex-api ALL DEPENDS ${MEX_OUTPUT})

	ADD_DEPENDENCIES(mex-api nemo)

	IF(WIN32)
		SET(INSTALL_DIR Matlab)
	ELSE(WIN32)
		SET(INSTALL_DIR share/nemo/matlab)
	ENDIF(WIN32)

	INSTALL(
		FILES
			${MEX_OUTPUT}
			nemo.m
			nemoNetwork.m
			nemoConfiguration.m
			nemoSimulation.m
			example.m
		DESTINATION
			${INSTALL_DIR}
	)

ELSE(NOT MEX STREQUAL MEX-NOTFOUND)
	MESSAGE("MEX compiler not found. Skipping Matlab API generation") 
ENDIF(NOT MEX STREQUAL MEX-NOTFOUND)
