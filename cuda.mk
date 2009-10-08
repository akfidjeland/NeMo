################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

CUDA_SDK_PATH := ext/NVIDIA_CUDA_SDK
SRCDIR := src/Simulation/CUDA/kernel/
ROOTDIR := dist/build/cuda

CU_MAIN = kernel_wrapper.cu

# CUDA source files which are #included in main kernel source file
CU_INC = kernel.cu L1SpikeQueue.cu firingProbe.cu partitionConfiguration.cu cycleCounting.cu error.cu connectivityMatrix.cu stdp.cu applySTDP.cu thalamicInput.cu

# CUDA source files (compiled with cudacc)
# CUFILES		:= $(addprefix $(SRCDIR),$(CU_MAIN) $(CU_INC))
CUFILES		:= $(addprefix $(SRCDIR),$(CU_MAIN))
CU_DEPS     := $(addprefix $(SRCDIR),$(CU_INC))
CCFILES     := $(addprefix $(SRCDIR),L1SpikeQueue.cpp FiringOutput.cpp RuntimeData.cpp ConnectivityMatrix.cpp time.cpp CycleCounters.cpp ThalamicInput.cpp RSMatrix.cpp StdpFunction.cpp)
CFILES		:= $(addprefix $(SRCDIR),kernel.c)


STATIC_LIB := libcuIzhikevich.a

EXTINCLUDES := -I/usr/include -I$(CUDA_SDK_PATH)/common/inc



################################################################################
# Rules and targets

#default: all

verbose = 1
include cuda_common.mk
NVCCFLAGS += --host-compilation=c++ --ptxas-options="-v" --maxrregcount 32 # --keep

################################################################################

ifeq ($(emu), 1)
	BINSUBDIR   := emu$(BINSUBDIR)
	# consistency, makes developing easier
	CXXFLAGS		+= -D__DEVICE_EMULATION__
	CFLAGS			+= -D__DEVICE_EMULATION__
endif
BINTARGETDIR := $(BINDIR)/$(BINSUBDIR)


cuda_count:
	@echo "Lines of kernel code: "
	@cat $(CUFILES) $(CCFILES) | wc -l 


################################################################################
# Profiling
################################################################################

PROFILE_DIR := profile

.PHONY: memprofile
memprofile: PATH := $(ABS_BIN_DIR):$(PATH)
memprofile: CUDA_PROFILE=1
memprofile: CUDA_PROFILE_CONFIG=profile.config 
memprofile: profile.config
	# TODO: complete
	#@(cd profile; random1k 
