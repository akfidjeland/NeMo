#!/bin/bash
#
# Usage: benchmark n_1 n_2 n_3
#
# where n_i is network size (in thousands)
#
# Results are printed to stdout

bm=../dist/build/benchmark/benchmark
log="err.log"

rm -f $log

# get output data file from command line?
# TODO: get synapse count from command-line

$bm --print-header
for n in $*; do
	$bm --neurons=$n --synapses=1000 2>> $log
done
