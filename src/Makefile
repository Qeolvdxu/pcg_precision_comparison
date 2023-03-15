CC = gcc
NVCC = nvcc

CFLAGS = -g -Wall -Wextra -pedantic -v -gdwarf-4
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wpedantic -x c -g -G -v

LIB_FLAGS = -lm -lcudart -lcusparse -lcuda -lcublas -lpthread

BUILDDIR = ./build/

TARGET = cgpc

all: $(TARGET)

$(TARGET): cudacode.o 
	$(CC) -o $(BUILDDIR)$(TARGET) $(CFLAGS) main.c my_crs_matrix.c CCG.c $(BUILDDIR)CuCG.o $(LIB_FLAGS)

cudacode.o:
	mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -o $(BUILDDIR)CuCG.o -c CuCG.cu

clean:
	rm -rf $(BUILDDIR)*.o $(BUILDDIR)$(TARGET)

#nvcc -O3 -x c -g -G -o CuCG.o CuCG.cu -lcusparse -lcudart -lcuda -lm
#nvcc -O3 CuCG.o -o dlink.o -dlink
#gcc -c main.c -o main.o -Wall
#gcc -o cgpc -Wall -lcuda -lcudart -lcusparse -lm dlink.o main.o CuCG.o

CCG_TEST_M:
	make -f CCG_TEST.mk

CuCG_TEST_M:
	make -f CuCG_TEST.mk
