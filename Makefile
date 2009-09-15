HASKELL_BUILD_DIR := dist/build
CUDA_LIB := dist/build/cuda/lib/libcuIzhikevich.a

all: cabal

version :=$(shell util/version)
machine :=$(shell uname -m)

# TODO: use configure for this
thrift_inc =/usr/local/include/thrift
thrift_dir = /usr/local/lib

thrift_build = dist/build/thrift



# Build haskell via 'make' to allow parallel build using the -j flag. This is
# passed on to the recursive make call. Cabal doesn't support building specific
# flags. Selecting targets is done via ./Setup.lhs configure.
cabal: cuda $(thrift_build)/gen-hs
	@echo "Using configuration from previous run of ./Setup.lhs configure"
	./Setup.lhs build


# CUDA kernel
.PHONY: cuda
cuda:
	make -f cuda.mk $(CUDA_LIB)


client: matlab-client

autogen := dist/build/autogen


matlab_src = src/ExternalClient/matlab
matlab_build = dist/build/matlab
matlab_m_files := $(basename $(basename $(notdir $(wildcard $(matlab_src)/*.m.m4))))



matlab-client: $(matlab_build)/nemo_mex.mexa64 \
	$(patsubst %,$(matlab_build)/%.m,$(matlab_m_files))


# Generate LUT for Matlab API function dispatch
# (m4-macros for m-file or c++ function array)
.PRECIOUS: $(autogen)/mex_fn_lut.hpp
$(autogen)/mex_fn_lut.%: $(matlab_src)/gen_fn.py $(matlab_src)/*.m.m4
	mkdir -p $(dir $@)
	$< --$* $(matlab_m_files) > $@


# Generate Matlab files from source, documentation, and function LUT
$(matlab_build)/%.m: $(autogen)/mex_fn_lut.m4 $(matlab_src)/%.m.m4 $(matlab_src)/%.m.help
	$(matlab_src)/m-help $(word 3, $^) > $@
	m4 $(wordlist 1, 2, $^) >> $@


# Generate Matlab reference source files from source and documentation
$(matlab_build)/reference/%.m: $(matlab_src)/%.m.ref $(matlab_src)/%.m.help
	mkdir -p $(dir $@)
	$(matlab_src)/m-help $(word 2, $^) > $@
	cat $< >> $@


# TODO: detect the default extension for matlab-mex
# TODO: windows build
$(matlab_build)/%.mexa64: $(matlab_src)/%.cpp $(thrift_build)/gen-cpp $(autogen)/mex_fn_lut.hpp
	mkdir -p $(dir $@)
	matlab-mex -I$(thrift_inc) -I$(thrift_build)/gen-cpp -I$(autogen) -lthrift \
		-o $(matlab_build)/$* \
		$< $(addprefix $(thrift_build)/gen-cpp/,NemoFrontend.cpp nemo_types.cpp)


.PRECIOUS: $(thrift_build)/gen-%
$(thrift_build)/gen-%: src/ExternalClient/nemo.thrift
	mkdir -p $(thrift_build)
	thrift --gen $* -o $(thrift_build) $<
	touch $@


# Target only needed for install
$(HASKELL_BUILD_DIR)/nemo-server/nemo-server: all


#
# Distribution
#

dist_dir := dist/build/nemo-$(machine)-$(strip $(version))
doc_build := dist/build/manual

.PHONY: dist
dist: $(dist_dir).zip


# TODO: get architecture from system
# TODO: use proper dependencies here
$(dist_dir).zip: $(doc_build)/manual.pdf client cabal
	mkdir -p $(basename $@)
	cp --target-directory $(dist_dir) -r $< dist/build/nemo/nemo dist/build/matlab
	strip $(dist_dir)/nemo
	# include shared thrift libraries as well.
	(cd dist/build; zip -r $(notdir $@) $(basename $(notdir $@)); cd ../..)

#
# Documentation
#


.PHONY: doc
doc: $(doc_build)/manual.pdf

# The wiki page includes just a single combined file
$(doc_build)/functionReference: $(matlab_src)/*.m.help
	mkdir -p $(dir $@)
	cat $^ > $@


$(doc_build)/manual.tex: doc/wiki/MatlabAPI \
		doc/manual/manual_stylesheet.tex \
		$(doc_build)/functionReference
	mkdir -p $(dir $@)
	cp --target-directory $(dir $@) $(wordlist 1,2,$^)
	rst2latex --use-latex-docinfo --use-latex-toc --no-section-numbering \
		--stylesheet=manual_stylesheet.tex \
		$(doc_build)/MatlabAPI > $@

$(doc_build)/manual.pdf: $(doc_build)/manual.tex
	(cd $(dir $<); pdflatex $(notdir $<); cd ..)


# Run after building with cabal
install: # $(HASKELL_BUILD_DIR)/nemo-server/nemo-server
	install $(HASKELL_BUILD_DIR)/nemo-server/nemo-server /usr/local/bin

.PHONY: tags
tags:
	xargs --arg-file=taggable hasktags --ctags --append
	#xargs --arg-file=taggable ghc -e :ctags --fno-warn-missing-modules
	#find -iname '*.hs' -or -iname '*.hsc' | xargs hasktags
	#find src -iname '*.hs' -or -iname '*.hsc' | xargs ghc -e :ctags
	#ghc -e :ctags --make src/NSim.hs

# There ought to be a better way of doing this!
.PHONY: relink
relink:
	rm -f dist/build/nemo/nemo
	./Setup.lhs build


.PHONY: count
count:
	@echo "Lines of haskell source code: "
	@find src -iname '*.hs' -or -iname '*.hsc' | xargs cat | grep -v -e '^[[:space:]]*$$' -e '^[[:space:]]*--' -e '^ - ' -e '^{- ' | wc -l
	@echo "Lines of test code: "
	@find testsuite/Test -iname '*.hs' -or -iname '*.hsc' | xargs cat | grep -v -e '^[[:space:]]*$$' -e '^[[:space:]]*--' -e '^ - ' -e '^{- ' | wc -l
	make -f cuda.mk count


# The CUDA lib is built in cabals build tree, so it will be cleaned out along
# with all the haskell files.
.PHONY: clean
clean:
	./Setup.lhs clean
