#!/bin/sh
#
# Generate LaTeX documentation for the Matlab API based on the in-file help
# strings. These are already in ReST format and can be processed with
# rst2latex. However, rst2latex produces a stand-alone document, so we need to
# strip out some crud before we can include this in another LaTeX document.
#
# This script reads from stdin and writes to stdout

rst2latex | sed -e '1,/begin{document}/d' -e '/end{document}/d' -e '/pdfbookmark/d' -e '/hypertarget/d' -e 's/^\\section/\\paragraph/'
