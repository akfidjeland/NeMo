#!/usr/bin/env python
#
# Usage:
#   gen_fn --m4
#   gen_fn --cpp

import sys

functions = ['connect', 'disconnect', 'setNetwork', 'run', 'applyStdp']


def generateFnArray():
    """ Generate a c function array for inclusion into nemo_mex """
    print "#define FN_COUNT", len(functions)
    print "typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);"
    print "fn_ptr fn_arr[FN_COUNT] = {\n\t",
    print ",\n\t".join(functions)
    print "};"


def generateM4Macros():
    for (i, fn) in enumerate(functions):
        print "define(mex_%s, uint32(%s))dnl" % (fn, i)


if __name__ == "__main__":
    output = sys.argv[1]
    if output == '--m4':
        generateM4Macros()
    elif output == '--cpp':
        generateFnArray()
