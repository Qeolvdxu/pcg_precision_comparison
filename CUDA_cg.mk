CC = nvcc
TARGET = CUDA_cg
SOURCE = CUDA_cg.cu
LFLAGS = -lcusparse
all:
	$(CC) -o CUDA_cg $(SOURCE) $(LFLAGS)
