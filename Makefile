HASKELL_BUILD_DIR := dist/build
CUDA_LIB := dist/build/cuda/lib/libcuIzhikevich.a

# Build haskell via make to allow parallel build using the -j flag. This is
# passed on to the recursive make call. Cabal doesn't support building specific
# flags. Selecting targets is done via ./Setup.lhs configure.
all: cuda haskell.mk
	@echo "Using configuration from previous run of ./Setup.lhs configure"
	make -f haskell.mk
	./Setup.lhs build


# This makefile relies on an existing file haskell.mk. This can be generated
# using Setup makefile -f haskell.mk. Make sure to specify the output file, as
# otherwise THIS file will be overwritten.
haskell.mk: nemo.cabal
	rm -f haskell.mk
	./Setup.lhs makefile -f $@


# CUDA kernel
.PHONY: cuda
cuda:
	make -f cuda.mk $(CUDA_LIB)


# Target only needed for install
$(HASKELL_BUILD_DIR)/nemo-server/nemo-server: all

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
	rm -f dist/build/ring/ring dist/build/smallworld/smallworld dist/build/random1k/random1k dist/build/runtests/runtests  dist/build/simple/simple dist/build/benchmark/benchmark dist/build/nemo-server/nemo-server
	#rm -f dist/build/smallworld/smallworld
	#rm -f dist/build/simple/simple
	#rm -f dist/build/ring/ring 
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
	rm -f haskell.mk
