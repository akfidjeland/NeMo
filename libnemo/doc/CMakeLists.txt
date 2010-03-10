##############################################################################
# API Documentation
##############################################################################

# Users shouldn't normally need to build to documentation themselves.
OPTION(REBUILD_API_DOCS "Rebuild API documentation" OFF)

SET(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doc)
SET(DOXYGEN_MAN_OUTPUT_DIR ${DOXYGEN_OUTPUT_DIR}/man/man3)
SET(DOXYGEN_MAN_OUTPUTS
	${DOXYGEN_MAN_OUTPUT_DIR}/nemo.h.3
	${DOXYGEN_MAN_OUTPUT_DIR}/nemo_Network.3
)

# Some extra pages, which should be cleaned but not installed
# TODO: add these to source package ignore files
SET(DOXYGEN_MAN_EXTRA_OUTPUTS
	${DOXYGEN_MAN_OUTPUT_DIR}/Network.hpp.3
)

SET(DOXYGEN_OUTPUTS
	${DOXYGEN_MAN_OUTPUTS}
	${DOXYGEN_MAN_EXTRA_OUTPUTS}
)

IF(REBUILD_API_DOCS)
	FIND_PACKAGE(Doxygen)

	IF(DOXYGEN_FOUND)

		SET(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
		CONFIGURE_FILE(${DOXYFILE_IN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ESCAPE_QUOTES @ONLY)

		# Run Doxygen
		# TODO: don't copy the whole doc-tree into the source tree. Most of
		# the generated files are not needed for the source installation.
		ADD_CUSTOM_COMMAND(
			OUTPUT ${DOXYGEN_OUTPUTS}
			COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYGEN_MAN_OUTPUT_DIR} ${CMAKE_SOURCE_DIR}/doc/man/man3
			DEPENDS
				${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
				${CMAKE_SOURCE_DIR}/src/nemo.h
				${CMAKE_SOURCE_DIR}/src/Network.hpp
		)

		ADD_CUSTOM_TARGET(api-doc
			DEPENDS ${DOXYGEN_OUTPUTS} ${DOXYGEN_SRC_OUTPUT}
			COMMENT "Generating man pages"
			VERBATIM
		)

		# Doxygen produces rather cruddy man-pages. We do some replacements to fix header etc.
		FIND_PROGRAM(SED sed)
		IF(SED)
			FOREACH(MANPAGE ${DOXYGEN_MAN_OUTPUTS})
				# TODO: get date from somewhere
				SET(SED_SET_HEADER -e s/.TH.*/.TH\ nemo.h\ 3\ \"Mar\ 2010\"\ \"\"\ \"Nemo\ Reference\ Manual\"/)
				SET(SED_SET_AUTHOR -e s/Generated\ automatically\ by\ Doxygen.*/Andreas\ Fidjeland\ \(using\ Doxygen\)/)
				# Add reference to main man-page and upper-case author header
				SET(SED_SET_SEE -e /.SH\ .Author/\ c.SH\ SEE\ ALSO\\nnemo\(3\)\ for\ library\ overview\\n.SH\ AUTHOR)
				ADD_CUSTOM_COMMAND(
					TARGET api-doc POST_BUILD
					COMMAND ${SED} -i ${SED_SET_HEADER} ${SED_SET_AUTHOR} ${SED_SET_SEE} ${MANPAGE}
					VERBATIM
				)
			ENDFOREACH(MANPAGE ${DOXYGEN_MAN_OUTPUTS})
		ENDIF(SED)

		# TODO: generate PDF manual

	ENDIF(DOXYGEN_FOUND)

ELSE(REBUILD_API_DOCS)

	# The generated documentation should already be present in the source tree.
	# Copy it into the build tree.
	ADD_CUSTOM_COMMAND(
		OUTPUT ${DOXYGEN_OUTPUTS}
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc ${CMAKE_BINARY_DIR}/doc
	)

	ADD_CUSTOM_TARGET(api-doc
		DEPENDS ${DOXYGEN_OUTPUTS} ${DOXYGEN_SRC_OUTPUT}
		COMMENT "Generating man pages"
		VERBATIM
	)

ENDIF(REBUILD_API_DOCS)


# Always create API documentation
ADD_DEPENDENCIES(nemo api-doc)


IF(UNIX)
	INSTALL(FILES ${DOXYGEN_MAN_OUTPUTS} DESTINATION man/man3)
	INSTALL(FILES man/man3/nemo.3 DESTINATION man/man3)
ENDIF(UNIX)
