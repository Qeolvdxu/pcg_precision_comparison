CC = nvcc
TARGET = cgcsrcuda
SOURCE = cgcsrcuda.cu
LFLAGS = -lcusparse
all:
	$(CC) -o cgcsrcuda $(SOURCE) $(LFLAGS)
