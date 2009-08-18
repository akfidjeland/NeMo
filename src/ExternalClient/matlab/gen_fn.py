#!/usr/bin/env python
#
# Usage:
#   gen_fn --m4
#   gen_fn --cpp

import sys


def fn_name(filename):
    """ convert from 'nemoXxxYyy' format to 'xxxYyy' format """
    return filename[4].lower() + filename[5:]


def generate_fn_array(functions):
    """ Generate a c function array for inclusion into nemo_mex """
    print "#define FN_COUNT", len(functions)
    print "typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);"
    print "fn_ptr fn_arr[FN_COUNT] = {\n\t",
    print ",\n\t".join(functions)
    print "};"


def generate_m4_macros(functions):
    for (i, fn) in enumerate(functions):
        print "define(mex_%s, uint32(%s))dnl" % (fn, i)


if __name__ == "__main__":
    output = sys.argv[1]
    functions = [fn_name(x) for x in sys.argv[2:]]
    if output == '--m4':
        generate_m4_macros(functions)
    elif output == '--hpp':
        generate_fn_array(functions)
