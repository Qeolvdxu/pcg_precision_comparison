CC = gcc
NVCC = nvcc

CFLAGS = -std=c99 -g -Wall -Wextra -pedantic -v
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wpedantic -x c -g -G -v

LIB_FLAGS = -lm -lcudart -lcusparse -lcuda -lcublas

BUILDDIR = ./build/

TARGET = cgpc

all: $(TARGET)

$(TARGET): cudacode.o 
	$(CC) -o $(TARGET) $(CFLAGS) main.c CuCG.o $(LIB_FLAGS)

cudacode.o:
	$(NVCC) $(NVCCFLAGS) -c CuCG.cu

clean:
	rm -rf *.o $(TARGET)

#nvcc -O3 -x c -g -G -o CuCG.o CuCG.cu -lcusparse -lcudart -lcuda -lm
#nvcc -O3 CuCG.o -o dlink.o -dlink
#gcc -c main.c -o main.o -Wall
#gcc -o cgpc -Wall -lcuda -lcudart -lcusparse -lm dlink.o main.o CuCG.o

CCG_TEST_M:
	make -f CCG_TEST.mk

CUDA_cg_M:
	make -f CuCG_TEST.mk
