SUBDIRS(man)
SUBDIRS(api_html)
SUBDIRS(dev_html)
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/manual/CMakeLists.txt)
	# No LaTeX build in the source package,
	SUBDIRS(manual)
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/manual/CMakeLists.txt)
