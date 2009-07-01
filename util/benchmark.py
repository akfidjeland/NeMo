#!/usr/bin/env python
import os
import subprocess
import sys
from socket import gethostname

benchmark_bin = "../dist/build/benchmark/benchmark"

def main(args):
    # first arg is p(local), while the rest are number of partitions
    # open files for writing or append (default)
    p_local = float(args[0])
    m = 1000 # TODO: get this from command-line
    f_dat = open(filename(m, p_local, 'dat'), 'a')
    #f_err = open(filename(m, p_local, 'err'), 'w')
    call_command(benchmark_bin + " --print-header", f_dat)
    for cc in args[1:]:
        print benchmark_cmd(cc, m, p_local)   
        call_command(benchmark_cmd(cc, m, p_local), f_dat)
    #f_err.close()
    f_dat.close()


def benchmark_cmd(cc, m, p_local):
    return "%s --neurons=%s --synapses=%s --local-probability=%s" % \
        (benchmark_bin, cc, m, p_local)


def filename(m, p_local, ext):
    """ return filename for current benchmark run """
    return "local%s.%s.%s.%s" % (int(p_local*100), gethostname(), m, ext)


def call_command(command, out):
    # TODO: no need for split again
    process = subprocess.Popen(command.split(' '), stdout=out)
    process.wait()
    return process.returncode

if __name__ == "__main__":
    main(sys.argv[1:])
