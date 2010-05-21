
OPTION(DOC_REBUILD_PDF_MANUAL "Rebuild API documentation (PDF manual)" OFF)

IF(DOC_REBUILD_PDF_MANUAL)

	FIND_PACKAGE(LATEX REQUIRED)

	ADD_CUSTOM_COMMAND(
		OUTPUT manual.pdf
		# TODO: use configure step to do this
		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/manual.tex ${CMAKE_CURRENT_BINARY_DIR}
		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/manual.bib ${CMAKE_CURRENT_BINARY_DIR}
		COMMAND ${PDFLATEX_COMPILER} manual.tex
		COMMAND ${BIBTEX_COMPILER} manual
		COMMAND ${PDFLATEX_COMPILER} manual.tex
		DEPENDS manual.tex
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)

	ADD_CUSTOM_TARGET(api-doc-latex
		DEPENDS manual.pdf
		COMMENT "Generating PDF manual"
		VERBATIM
	)

	# Copy final generated files into source directory, for inclusion in source package
	ADD_CUSTOM_COMMAND(
		TARGET api-doc-latex POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy manual.pdf ${CMAKE_CURRENT_SOURCE_DIR}
	)

ELSE(DOC_REBUILD_PDF_MANUAL)

	# The generated documentation should already be present in the source tree.
	# Copy it into the build tree.
	ADD_CUSTOM_COMMAND(
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/manual.pdf
		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/manual.pdf ${CMAKE_CURRENT_BINARY_DIR}
	)

	ADD_CUSTOM_TARGET(api-doc-latex
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/manual.pdf
		COMMENT "Using existing manual"
		VERBATIM
	)
ENDIF(DOC_REBUILD_PDF_MANUAL)

# Always create API documentation
ADD_DEPENDENCIES(nemo api-doc-latex)

IF(UNIX)
	INSTALL(FILES manual.pdf DESTINATION share/doc/nemo)
ENDIF(UNIX)

IF(WIN32)
	INSTALL(FILES manual.pdf DESTINATION .)
ENDIF(WIN32)
