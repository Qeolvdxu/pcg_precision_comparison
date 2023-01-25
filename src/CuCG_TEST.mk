CC = nvcc

CFLAGS = -Wall
LDFLAGS = -lcusparse

TARGET = CuCU_TEST
SOURCE = CuCG_TEST.cu CuCG.cu my_crs_matrix.c


all:
	$(CC) $(CFLAGS) -o CuCG $(SOURCE) $(LDFLAGS)
