CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(NeMo)

SET(CPACK_PACKAGE_NAME nemo)
SET(CPACK_PACKAGE_VERSION_MAJOR 0)
SET(CPACK_PACKAGE_VERSION_MINOR 6)
SET(CPACK_PACKAGE_VERSION_PATCH 9)
SET(CPACK_PACKAGE_CONTACT "Andreas Fidjeland <andreas.fidjeland@imperial.ac.uk>")
SET(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_SOURCE_DIR}/description.txt)

SET(NEMO_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

SUBDIRS(src doc)

# read description from file
FILE(READ ${CPACK_PACKAGE_DESCRIPTION_FILE} NEMO_DESCRIPTION)

IF(APPLE OR WIN32)
	# PackageMaker cannot handle files without an extension.
	# Windows won't fall over, but probably a good idea there as well.
	SET(README "ReadMe.txt")
	SET(COPYING "Copying.txt")
	SET(INSTALLATION "Install.txt")
ELSE(APPLE OR WIN32)
	SET(README "README")
	SET(COPYING "COPYING")
	SET(INSTALLATION "INSTALL")
ENDIF(APPLE OR WIN32)

CONFIGURE_FILE(README.in ${CMAKE_BINARY_DIR}/${README} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(README.in ${CMAKE_SOURCE_DIR}/${README} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(COPYING ${CMAKE_BINARY_DIR}/${COPYING} ESCAPE_QUOTES @ONLY)
CONFIGURE_FILE(INSTALL ${CMAKE_BINARY_DIR}/${INSTALLATION} ESCAPE_QUOTES @ONLY)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Packaging

SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/${COPYING}")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/${README}")

IF(WIN32)
	SET(CPACK_GENERATOR "NSIS")
	# Not that for the path setting to work, the libraries *must* be in the bin
	# folder, as CMake is hard-coded this way.
	SET(CPACK_NSIS_MODIFY_PATH ON)
	# By default CMake puts the installed version into a directory which
	# includes the version in the name. We can reasonably assume that users
	# won't have more than a single version installed at a time, so always
	# install into the same directory, to make upgrades easier.
	SET(CPACK_PACKAGE_INSTALL_DIRECTORY "NeMo")
	# TODO: Could add links to manual here using CPACK_NSIS_MENU_LINKS. Did not immideatly work
ENDIF(WIN32)

IF(APPLE)
	SET(CPACK_GENERATOR "PackageMaker")
ENDIF(APPLE)

SET(CPACK_SOURCE_GENERATOR "TGZ" "ZIP")

SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${NEMO_VERSION}")
SET(CPACK_SOURCE_IGNORE_FILES
	"\\\\.gitignore"
	".git"
	".later"
	".*swp$"
	release_notes
	build
	src/build
	src/mpi
	src/api/autogen
	src/api/matlab/sources
	src/api/matlab/examples
	src/api/python/utils
	src/gui
	src/nemo/remote2
	doc/wiki
	doc/manual
	util
	benchmarking
	dsl
	README.in # only include the generated file
	ReadMe.txt # for PackageMaker only
	"tmp.*"
	${CPACK_SOURCE_IGNORE_FILES})

INCLUDE(CPack)
