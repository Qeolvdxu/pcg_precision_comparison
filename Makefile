CC = gcc
NVCC = nvcc

CFLAGS = -O3 -Wall -Wextra -pedantic -gdwarf-4 #-DINJECT_ERROR
NVCCFLAGS = -O3 -Wextra -Wall -Wpedantic #-DINJECT_ERROR

# Retrieve the value of the 'gpu' variable from the command line
GPU_MODE := $(gpu_mode)
ifeq ($(GPU_MODE),debug)
    NVCCFLAGS += -DENABLE_TESTS -g
endif
CPU_MODE := $(cpu_mode)
ifeq ($(CPU_MODE),debug)
    CFLAGS += -DENABLE_TESTS -g
endif


# Retrieve the value of the 'gpu' variable from the command line
GPU_PRECI ?=single
# Check the value of the 'gpu' variable and add corresponding compiler flags
ifeq ($(GPU_PRECI),single)
    NVCCFLAGS += -DCUDA_SINGLE
    CFLAGS += -DCUDA_SINGLE
endif
ifeq ($(GPU_PRECI),double)
    NVCCFLAGS += -DCUDA_DOUBLE
    CFLAGS += -DCUDA_DOUBLE
endif

# Retrieve the value of the 'c' variable from the command line
CPU_PRECI ?=double
# Check the value of the 'c' variable and add corresponding compiler flags
ifeq ($(CPU_PRECI),single)
    CFLAGS += -DC_SINGLE
endif
ifeq ($(CPU_PRECI),double)
    CFLAGS += -DC_DOUBLE
endif

LIB_FLAGS = -lm -lcudart -lcusparse -lcuda -lcublas -lpthread -fopenmp

SRCDIR = ./src/
BUILDDIR = ./Build/

TARGET = cgpc

all: $(TARGET)

$(TARGET): $(BUILDDIR)cudacode.o 
	$(CC) -o $(BUILDDIR)$(TARGET) $(CFLAGS) $(SRCDIR)main.c $(SRCDIR)my_crs_matrix.c $(SRCDIR)trisol.c $(SRCDIR)CCG.c $(BUILDDIR)CuCG.o $(LIB_FLAGS)

$(BUILDDIR)cudacode.o:
	mkdir -p $(BUILDDIR)
	$(CC) $(NVCCFLAGS) -o $(BUILDDIR)CuCG.o -c $(SRCDIR)CuCG.c -lcusparse

clean:
	rm -rf $(BUILDDIR)*.o $(BUILDDIR)$(TARGET)

CCG_TEST_M:
	make -f CCG_TEST.mk

CuCG_TEST_M:
	make -f CuCG_TEST.mk

