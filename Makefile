CC = gcc
NVCC = nvcc

CFLAGS = -Iinclude -O3 -Wall -Wextra -pedantic $(CONFIG)

LIB_FLAGS = -lm -lcudart -lcusparse -lcuda -lcublas -lpthread -fopenmp

SRCDIR = ./src/
BUILDDIR = ./Build/

TARGET = cgpc

all: $(TARGET)

$(TARGET): $(BUILDDIR)cudacode.o 
	$(CC) -o $(BUILDDIR)$(TARGET) $(CFLAGS) $(SRCDIR)main.c $(SRCDIR)my_crs_matrix.c $(SRCDIR)trisol.c $(SRCDIR)CCG.c $(BUILDDIR)CuCG.o $(LIB_FLAGS)

$(BUILDDIR)cudacode.o:
	mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -o $(BUILDDIR)CuCG.o -c $(SRCDIR)CuCG.c -lcusparse

clean:
	rm -rf $(BUILDDIR)*.o $(BUILDDIR)$(TARGET)

CCG_TEST_M:
	make -f CCG_TEST.mk

CuCG_TEST_M:
	make -f CuCG_TEST.mk

