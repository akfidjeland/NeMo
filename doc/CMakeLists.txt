SUBDIRS(man)
SUBDIRS(api_html)
SUBDIRS(dev_html)
IF(EXISTS manual/CMakeLists.txt)
	# No LaTeX build in the source package,
	SUBDIRS(manual)
ENDIF(EXISTS manual/CMakeLists.txt)
