CC = gcc
NVCC = nvcc

CFLAGS = -g -Wall -Wextra -pedantic -v -gdwarf-4
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wpedantic -x c -g -G -v

LIB_FLAGS = -lm -lcudart -lcusparse -lcuda -lcublas

BUILDDIR = ./build/

TARGET = CuCG_TEST 

all: $(TARGET)

$(TARGET): cudacode.o 
	$(CC) -o $(BUILDDIR)$(TARGET) $(CFLAGS) CuCG_TEST.c $(BUILDDIR)CuCG.o $(LIB_FLAGS)

cudacode.o:
	mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -o $(BUILDDIR)CuCG.o -c CuCG.cu
