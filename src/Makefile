aSM := 35

CC := gcc
NVCC := nvcc

CFLAGS = -std=c99
NVCCFLAGS = -O3

LIB_FLAGS = -L/usr/local/lib -lcudadevrt -lcudart -lcusparse -lcuda

BUILDDIR = build

TARGET = cgpc
all: $(TARGET)

$(TARGET): $(BUILDDIR)/dlink.o $(BUILDDIR)/main.o $(BUILDDIR)/$(TARGET).o
	$(CC)  $(CFLAGS)  -o $@ $(LIB_FLAGS) $^
	echo 1

$(BUILDDIR)/dlink.o: $(BUILDDIR)/$(TARGET).o 
	$(NVCC)  $(NVCCFLAGS) $^ -o $@ $(LIB_FLAGS) -dlink
	echo 2

$(BUILDDIR)/main.o: main.c
	$(CC) $(CFLAGS) $(LIB_FLAGS) -c $< -o $@
	echo 3
$(BUILDDIR)/$(TARGET).o: CuCG.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@ $(LIB_FLAGS)
	echo 4

clean:
	rm -f $(BUILDDIR)/*.o $(TARGET) ll:
#nvcc -O3 -x c -g -G -o CuCG.o CuCG.cu -lcusparse -lcudart -lcuda -lm
#nvcc -O3 CuCG.o -o dlink.o -dlink
#gcc -c main.c -o main.o -Wall
#gcc -o cgpc -Wall -lcuda -lcudart -lcusparse -lm dlink.o main.o CuCG.o

CCG_TEST_M:
	make -f CCG_TEST.mk

CUDA_cg_M:
	make -f CuCG_TEST.mk
