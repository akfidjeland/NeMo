pdf = basic.pdf pipelined_thrift.pdf pipelined_all.pdf

all: $(pdf)

basic.dot: rpc.m4.dot
	m4 $< > $@

pipelined_thrift.dot: rpc.m4.dot
	m4 -Dpipeline_thrift $< > $@

pipelined_all.dot: rpc.m4.dot
	m4 -Dpipeline_backend -Dpipeline_thrift $< > $@

%.pdf: %.dot
	dot -Tpdf $< -o $@

.PHONY: clean
clean: 
	rm -f $(pdf)

