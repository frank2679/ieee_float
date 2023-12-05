SRC = main.cpp

# Env check routines
ifneq ("$(wildcard $(BRCC_PATH)/bin/brcc)","")
BRCC_CLANG=1
endif

ifneq ("$(wildcard ${BESU_PATH}/lib/libbesu.so)","")
BESU_LIB=1
endif

ifneq ("$(wildcard ${SUPA_PATH}/include/supa.h)","")
SUPA_HEADER=1
endif

# BRCC+BESU checking
ifdef CHECK_SUPA_ENV
check_br_env =\
  $(if $(value BRCC_PATH),, \
      $(error BRCC_PATH Undefined)) \
  $(if $(value BRCC_CLANG),, \
      $(error BRCC brcc cannot be found!)) \
  $(if $(value BESU_PATH),, \
      $(error BESU_PATH Undefined)) \
  $(if $(value BESU_LIB),, \
      $(error BESU libbesu.so cannot be found!)) \
  $(if $(value SUPA_PATH),, \
      $(error SUPA_PATH Undefined)) \
  $(if $(value SUPA_HEADER),, \
      $(error SUPA header files cannot be found!))
endif

# Executable name

ifeq ($(PROG),)
PROG := $(notdir $(shell pwd))
endif

BESU_INCLUDE_PATH=${BESU_PATH}/include
SUPA_INCLUDE_PATH=${SUPA_PATH}/include
SUPA_HELPER_INCLUDE_PATH=${BASE_PATH}/common/include

# For BRCC compiling
ifdef BRCC_PATH
BRCC=${BRCC_PATH}/bin/brcc
else
BRCC=brcc
endif

# Optimizer level
ifdef OPTNONE
  OPTFLAG=-O0
else
  OPTFLAG=-O2
endif

# Debug info
ifdef DEBUG
  DEBUGFLAG=-g
  OPTFLAG=-O0
endif

ifdef SUPA_ARCH_VERSION
VERSION=${SUPA_ARCH_VERSION}
endif

BIREN_ERRATA_FLAG=-D__BR10X_ERRATA_06_GLOBAL_TIME__ -D__BR10X_ERRATA_41_ADD_MATRIX_ROW_H_CONVWEIGHT_ALIGNMENT_OCH__ -D__BR10X_ERRATA_37_POOL_IDX_PADDING__
BIREN_ERRATA_FLAG += -D__BR10X_ERRATA_43_GEMM_REDUCE_TO_ROW_MAJOR_BLOCK__

OPTFLAG+=-march=br100,br110 # support BR100, BR100

ifeq (${VERSION}, 110)
	CUDA_BIREN_FLAG=-DCUDA_SUPA_ARCH=110
endif

CPPFLAGS+= -x supa -fPIC ${OPTFLAG} ${DEBUGFLAG} ${BIREN_ERRATA_FLAG} -I${SUPA_HELPER_INCLUDE_PATH}

# For SUPA CUDA BE compiling
NVCC=nvcc
CUDA_CPPFLAGS= -x cu -DSUPA_CUDA_BE -MD  --gpu-architecture compute_70 -dc ${BIREN_ERRATA_FLAG} ${CUDA_BIREN_FLAG}
ifdef DEBUG
	CUDA_CPPFLAGS+= -g
endif
# $CUDA_PROG_CPPFLAGS is supplied by each program's flag
CUDA_CPPFLAGS += -Xcompiler -fPIC -I${SUPA_INCLUDE_PATH}/cuda-be ${CUDA_PROG_CPPFLAGS}
CUDA_CPPFLAGS += -I${SUPA_HELPER_INCLUDE_PATH}

ifdef SUPA_PATH
CUDA_LINK_FLAGS += --gpu-architecture compute_70 -lsupa-cuda  -lsupa-cuda-runtime -L${SUPA_PATH}/lib
else
CUDA_LINK_FLAGS += --gpu-architecture compute_70 -lsupa-cuda  -lsupa-cuda-runtime -L/usr/lib
endif

BRCC_PROG_FLAGS += ${PROG_FLAGS}
CUDA_PROG_FLAGS += ${PROG_FLAGS}

# Rules for normal building
%.o: %.cpp
	@:$(call check_br_env)
	${BRCC} ${CPPFLAGS} ${BRCC_PROG_FLAGS} $< -c -o $@

%.o: %.su
	@:$(call check_br_env)
	${BRCC} ${CPPFLAGS} ${BRCC_PROG_FLAGS} $< -c -o $@

%.cuda.o: %.cpp
	${NVCC} ${CUDA_CPPFLAGS} ${CUDA_PROG_FLAGS} $< -o $@

%.cuda.o: %.su
	${NVCC} ${CUDA_CPPFLAGS} ${CUDA_PROG_FLAGS} $< -o $@

# Default case, only run BRCC compilation and generate LLVM
all: ${SRC}
	@:$(call check_br_env)
	@echo [Info] Check SUPA source ${SRC} with BRCC front-end and emit LLVM IR
	${BRCC} ${CPPFLAGS} -S -emit-llvm $^

supa: ${PROG}-supa.out


run-supa:  ${PROG}-supa.out
	./$<

# test and run-test are aliases of supa and run-supa for compiler team test
test: supa


run-test: run-supa
	
BRCC_PROG_LINKFLAGS += ${PROG_LINKFLAGS}

OBJS = $(SRC:.cpp=.o)

${PROG}-supa.out: ${OBJS}
	${BRCC} --supa-link ${OPTFLAG} ${DEBUGFLAG} ${OBJS} ${SUPA_LINK_FLAGS} ${CUDA_PROG_LINKFLAGS} -o $@

quick: ${SRC}
	${BRCC} ${OPTFLAG} ${SUPA_LINK_FLAGS} ${BRCC_PROG_LINKFLAGS} ${CPPFLAGS}  $^ -o ${PROG}-supa.out

# cuda target tries to build a CUDA BE executable
cuda: ${PROG}-cuda.out

run-cuda: ${PROG}-cuda.out
	./$<

CUDA_OBJS = $(SRC:.cpp=.cuda.o)

CUDA_PROG_LINKFLAGS += ${PROG_LINKFLAGS}

# Only generate object file, not linking. Used to validate code.
cuda-fe: ${CUDA_OBJS}
	

${PROG}-cuda.out: ${CUDA_OBJS}
	${NVCC} ${CUDA_LINK_FLAGS} ${CUDA_OBJS} ${CUDA_PROG_LINKFLAGS} -o $@


clean:
	rm -f *.o *.d *.ll *.a *.so *.out *.host *.supafb
	rm -fR q?p?k?_*.dat q?p?k?_*.json q?p?k?_*.elf tmp_biren #Clean driver dump

help:
	@echo Help of SUPA Makefile based building system
	@echo "Common Targets"
	@echo "    help: show this help;"
	@echo "    clean: make clean the current folder and all sub-folders"
	@echo "Targets for examples"
	@echo "    all: Use BRCC front-end to compile supa example and generate LLVM IR;"
	@echo "    supa: Use BRCC to compile and build supa example;"
	@echo "    run-supa: Use BRCC to compile and run supa example with BESU;"
	@echo "    cuda-fe: Use CUDA-BE/NVCC front-end to do compile test of supa example;"
	@echo "    cuda: Use CUDA-BE/NVCC to compile and build supa example;"
	@echo "    run-cuda: Use CUDA-BE/NVCC to compile and run supa example;"

.PHONY: clean help cuda-libs

# Automatically detect header file changes
-include $(OBJS:.o=.d)
-include $(CUDA_OBJS:.o=.d)
