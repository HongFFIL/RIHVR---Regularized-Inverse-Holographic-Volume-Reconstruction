
include ./findcudalib.mk

# -------------------------------------------------
# Set paths for CUDA and OpenCV libs
# -------------------------------------------------
CUDA_PATH ?= "/usr/local/cuda"
OPENCV_PATH ?= "/usr/include/opencv4"

# getting the CUDA_Version from the shell
CUDA_VERSION=$(shell $(NVCC) --version | grep release | cut -d ' ' -f 5 | cut -d ',' -f 1)
$(info $$CUDA_VERSION is [${CUDA_VERSION}])

# setting the hostcompiler
HOST_COMPILER := g++-9

# -------------------------------------------------
# Compiler flags
# -------------------------------------------------
NVCCFLAGS   := -m${OS_SIZE} -std=c++17 -ccbin $(HOST_COMPILER)
LDFLAGS     := -rpath $(CUDA_PATH)/lib64 -rpath "/usr/local/lib/" -rpath "/usr/local/lib64/"

# set flags for fastmath and debugging
fastmath = 1 
dbg = 0

EXTRA_NVCCFLAGS   ?=
ifeq ($(fastmath),1)
	EXTRA_NVCCFLAGS += -use_fast_math
endif

ifeq ($(dbg),1)
	NVCCFLAGS += -g -G
	TARGET := debug
else
	TARGET := release
endif

ALL_CCFLAGS := -DLINUX_BUILD
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_CCFLAGS += -w -Xcompiler -Wno-format-overflow

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


INCLUDES := -I../../common/inc -I$(CUDA_PATH)/samples/common/inc/ -I$(OPENCV_PATH)

LIBRARIES := -L. -lcufft -lcurand -lnppc -lnppial -lnppist -lnpps \
             -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

GENCODE_FLAGS := \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_75,code=compute_75 \
  -gencode arch=compute_86,code=compute_86 \
  -gencode arch=compute_86,code=sm_86
  
EXTRA_SRC := umnholo.h

# -------------------------------------------------
# Object files
# -------------------------------------------------
OBJ_ALL := compressive_holo.o blob3d.o cumat.o deconvolution.o hologram.o \
           holo_sequence.o holo_ui.o object_cloud.o optical_field.o \
           focus_metric.o particle_extraction.o point_cloud.o reconstruction.o \
           sparse_compressive_holo.o sparse_volume.o sparse_gradient.o \
           sparse_segmentation.o point_cloud_3d.o

CCBG_OBJ := ccbg.o
PARTICLETRACKING_OBJ := particletracking.o tracking.o

CUDA_TARGETS := make-background segmentation sparse-inverse-recon
CPP_TARGETS  := ccbg particletracking
TARGETS      := $(CUDA_TARGETS) $(CPP_TARGETS)

# -------------------------------------------------
# Targets
# -------------------------------------------------
all: segmentation make-background sparse-inverse-recon ccbg particletracking post

# -------------------------------------------------
# CUDA object rules
# -------------------------------------------------
%.o: %.cu $(EXTRA_SRC)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# Explicit rule for holo_ui.cpp 
holo_ui.o: holo_ui.cpp holo_ui.h $(EXTRA_SRC)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# -------------------------------------------------
# C++ object rules
# -------------------------------------------------
ccbg.o: ccbg.cpp
	$(HOST_COMPILER) -std=c++17 -O3 -I$(OPENCV_PATH) -c $< -o $@

particletracking.o: particletracking.cpp
	$(HOST_COMPILER) -std=c++17 -O3 -c $< -o $@

tracking.o: tracking.cpp
	$(HOST_COMPILER) -std=c++17 -O3 -c $< -o $@

# -------------------------------------------------
# CUDA executables
# -------------------------------------------------
make-background: main_create_background.o $(OBJ_ALL)
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARY_PATH) $(LIBRARIES)

sparse-inverse-recon: main_sparse_inverse.o $(OBJ_ALL)
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARY_PATH) $(LIBRARIES)

segmentation: main_segmentation.o $(OBJ_ALL)
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARY_PATH) $(LIBRARIES)

# -------------------------------------------------
# Pure C++ executables
# -------------------------------------------------
ccbg: $(CCBG_OBJ)
	$(HOST_COMPILER) -O3 -o $@ $^ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

particletracking: $(PARTICLETRACKING_OBJ)
	$(HOST_COMPILER) -O3 -o $@ $^

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
clean:
	rm -f *.o
	rm -rf lib

# -------------------------------------------------
# Post-build step 
# -------------------------------------------------
post:
	@echo "Moving executables to lib/ and cleaning .o files..."
	@mkdir -p lib
	@mv $(TARGETS) lib/
	@rm -f *.o
	@echo "Build complete."

.PHONY: all clean post
