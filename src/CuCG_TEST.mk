CC = nvcc

CFLAGS = -Wall
LDFLAGS = -lcusparse

TARGET = CuCU_TEST
SOURCE = CuCG_TEST.cu CuCG.cu my_cr

all:
	$(CC) $(CFLAGS) -o CUDA_cg $(SOURCE) $(LDFLAGS)
