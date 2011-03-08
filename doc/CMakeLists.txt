##############################################################################
# API Documentation
##############################################################################

# Users shouldn't normally need to build the documentation themselves.
OPTION(NEMO_DOC_MAN_REBUILD "Rebuild API documentation (man pages)" OFF)
OPTION(NEMO_DOC_SOURCE_ENABLED "Create source code documentation using Doxygen" OFF)

IF(NEMO_DOC_SOURCE_ENABLED)
	SET(NEMO_DOC_SOURCE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/doc/html" CACHE STRING "Directory in which to place generated source code documentation")
ENDIF(NEMO_DOC_SOURCE_ENABLED)

SUBDIRS(latex api_html)

SET(DOC_DIR ${CMAKE_BINARY_DIR}/doc)
SET(MAN_DIR ${DOC_DIR}/man/man3)
SET(GENERATED_MAN_PAGES
	${MAN_DIR}/nemo-c.3
	${MAN_DIR}/nemo-cpp.3
)


IF(NEMO_DOC_MAN_REBUILD OR NEMO_DOC_SOURCE_ENABLED)
	FIND_PACKAGE(Doxygen REQUIRED)
ENDIF(NEMO_DOC_MAN_REBUILD OR NEMO_DOC_SOURCE_ENABLED)


IF(NEMO_DOC_MAN_REBUILD)
	# TODO: can remove this conditional
	IF(DOXYGEN_FOUND)

		SET(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/man.doxy.in)
		CONFIGURE_FILE(${DOXYFILE_IN} ${CMAKE_CURRENT_BINARY_DIR}/man.doxy ESCAPE_QUOTES @ONLY)

		# Run Doxygen
		ADD_CUSTOM_COMMAND(
			OUTPUT
				# Files we are interested in
				${MAN_DIR}/nemo.h.3
				${MAN_DIR}/nemo_Network.3
				# Extra crud
				${MAN_DIR}/nemo.hpp.3
			COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/man.doxy
			DEPENDS
				${CMAKE_CURRENT_BINARY_DIR}/man.doxy
				${CMAKE_SOURCE_DIR}/nemo.h
				${CMAKE_SOURCE_DIR}/nemo.hpp
		)

		# Name C API man page correctly
		ADD_CUSTOM_COMMAND(
			OUTPUT ${MAN_DIR}/nemo-c.3
			COMMAND ${CMAKE_COMMAND} -E copy ${MAN_DIR}/nemo.h.3 ${MAN_DIR}/nemo-c.3
			DEPENDS ${MAN_DIR}/nemo.h.3
		)

		# Name C++ API man page correctly
		ADD_CUSTOM_COMMAND(
			OUTPUT ${MAN_DIR}/nemo-cpp.3
			COMMAND ${CMAKE_COMMAND} -E copy ${MAN_DIR}/nemo_Network.3 ${MAN_DIR}/nemo-cpp.3
			DEPENDS ${MAN_DIR}/nemo_Network.3
		)

		ADD_CUSTOM_TARGET(api-doc-man
			DEPENDS ${GENERATED_MAN_PAGES}
			COMMENT "Generating man pages for API"
			VERBATIM
		)

		FIND_PROGRAM(SED sed)
		FOREACH(MANPAGE ${GENERATED_MAN_PAGES})

			# Doxygen produces rather cruddy man-pages. We do some replacements to fix header etc.
			IF(SED)
				# TODO: get date from somewhere
				STRING(REGEX REPLACE "${MAN_DIR}/(.*).3" "\\1" MANPAGE_TITLE ${MANPAGE})
				SET(SED_SET_HEADER -e s/^\\.TH.*/.TH\ ${MANPAGE_TITLE}\ 3\ \"Mar\ 2010\"\ \"\"\ \"Nemo\ Reference\ Manual\"/)
				SET(SED_SET_AUTHOR -e s/Generated\ automatically\ by\ Doxygen.*/Andreas\ Fidjeland\ \(using\ Doxygen\)/)

				# The title page is incorrect as it referes to filename or
				# class. Just hard-code the relevant replacements.
				SET(SED_SET_TITLE_C -e s/^nemo.h\ \\\\-/${MANPAGE_TITLE}\ \\\\-/)
				SET(SED_SET_TITLE_CPP -e s/^nemo::Network\ \\\\-/${MANPAGE_TITLE}\ \\\\-/)
				SET(SED_SET_TITLE ${SED_SET_TITLE_C} ${SED_SET_TITLE_CPP})

				# Add reference to main man-page and upper-case author header
				SET(SED_SET_SEE -e /.SH\ .Author/\ c.SH\ SEE\ ALSO\\nnemo\(3\)\ for\ library\ overview\\n.SH\ AUTHOR)

				ADD_CUSTOM_COMMAND(
					TARGET api-doc-man POST_BUILD
					COMMAND ${SED} -i ${SED_SET_HEADER} ${SED_SET_AUTHOR} ${SED_SET_TITLE} ${SED_SET_SEE} ${MANPAGE}
					VERBATIM
				)
			ENDIF(SED)

			# Copy final generated files into source directory, for inclusion in source package
			ADD_CUSTOM_COMMAND(
				TARGET api-doc-man POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy ${MANPAGE} ${CMAKE_SOURCE_DIR}/doc/man/man3
			)

		ENDFOREACH(MANPAGE ${GENERATED_MAN_PAGES})

	ENDIF(DOXYGEN_FOUND)

ELSE(NEMO_DOC_MAN_REBUILD)

	# The generated documentation should already be present in the source tree.
	# Copy it into the build tree.
	ADD_CUSTOM_COMMAND(
		OUTPUT ${GENERATED_MAN_PAGES}
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/man/man3 ${CMAKE_BINARY_DIR}/doc/man/man3
	)

	ADD_CUSTOM_TARGET(api-doc-man
		DEPENDS ${GENERATED_MAN_PAGES}
		COMMENT "Using existing man pages"
		VERBATIM
	)

ENDIF(NEMO_DOC_MAN_REBUILD)


# Always create API man pages
ADD_DEPENDENCIES(nemo api-doc-man)


IF(UNIX)
	SET(MAN_INSTALL_DIR share/man/man3)
	INSTALL(FILES ${GENERATED_MAN_PAGES} DESTINATION ${MAN_INSTALL_DIR})
	INSTALL(FILES man/man3/nemo.3 DESTINATION ${MAN_INSTALL_DIR})
ENDIF(UNIX)


# The source code documentation is not included in either the source or binary
# packages because:
#
# 1. it is quite large (>6MB)
# 2. it is only useful to people who want to hack the internals of NeMo. Such
#    people would compile from source, so can generate this documentation as
#    needed.
IF(NEMO_DOC_SOURCE_ENABLED)

	SET(DOXYGEN_INPUT
		${CMAKE_SOURCE_DIR}/nemo.h
		${CMAKE_SOURCE_DIR}/nemo.hpp
		${CMAKE_SOURCE_DIR}/nemo
		${CMAKE_SOURCE_DIR}/nemo/cpu
		${CMAKE_SOURCE_DIR}/nemo/cuda
		${CMAKE_SOURCE_DIR}/nemo/network
		${CMAKE_SOURCE_DIR}/nemo/network/programmatic
	)

	# Convert dependency list to whitespace separated string
	SET(DOXYGEN_INPUT_STR "")
	FOREACH(ARG ${DOXYGEN_INPUT})
		SET(DOXYGEN_INPUT_STR "${DOXYGEN_INPUT_STR} ${ARG}")
	ENDFOREACH(ARG ${DOXYGEN_INPUT})

	# Fill in name, version number etc.
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/source.doxy.in
		${CMAKE_CURRENT_BINARY_DIR}/source.doxy
		ESCAPE_QUOTES @ONLY
	)

	# Doxygen creates a very large number of output files. List just the
	# top-level one.
	SET(DOXYGEN_OUTPUT ${NEMO_DOC_SOURCE_OUTPUT_DIR}/index.html)

	# Run Doxygen
	ADD_CUSTOM_COMMAND(
		OUTPUT ${DOXYGEN_OUTPUT}
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/source.doxy
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/source.doxy ${DOXYGEN_INPUT}
	)

	# Need a named target in order for dependencies to work out
	ADD_CUSTOM_TARGET(api-doc-source
		DEPENDS ${DOXYGEN_OUTPUT}
		COMMENT "Generating source code documentation"
		VERBATIM
	)

	# If enabled in configuration, always create source code docs
	ADD_DEPENDENCIES(nemo api-doc-source)

ENDIF(NEMO_DOC_SOURCE_ENABLED)
