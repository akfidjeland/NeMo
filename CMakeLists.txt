CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(NeMo)

SET(CPACK_PACKAGE_NAME nemo)
SET(CPACK_PACKAGE_VERSION_MAJOR 0)
SET(CPACK_PACKAGE_VERSION_MINOR 6)
SET(CPACK_PACKAGE_VERSION_PATCH 8)
SET(CPACK_PACKAGE_CONTACT "Andreas Fidjeland <andreas.fidjeland@imperial.ac.uk>")

SET(NEMO_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

SUBDIRS(src doc)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# Packaging

SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/INSTALL")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")

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
	# PackageMaker needs resource files to have .txt extension. These are just symlinks.
	SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/INSTALL.txt")
	SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING.txt")
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
	COPYING.txt INSTALL.txt # symlinks
	"tmp.*"
	${CPACK_SOURCE_IGNORE_FILES})

INCLUDE(CPack)
