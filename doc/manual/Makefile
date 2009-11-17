#manual.pdf: manual.tex figures/stdp/symmetric.pdf title.tex matlab-fref.tex 
manual.pdf: manual.tex \
		manual.bib \
		figures/rpc/basic.pdf \
		figures/rpc/pipelined_thrift.pdf
	pdflatex $<
	bibtex manual
	pdflatex $<

.PHONY: figures/rpc/*.pdf

figure/rpc/basic.pdf:
	make -C $(dir $@) $(notdir $@)

figure/rpc/pipelined_thrift.pdf:
	make -C $(dir $@) $(notdir $@)

clean:
	rm *.bbl *.aux *.log
	rm -f manual.pdf
