CC = nvcc

CFLAGS = -Xcompiler -Wall,-Wpedantic -x c -g -G 
LDFLAGS = -lcusparse -lcudart -lcuda

TARGET = CuCU_TEST
SOURCE = CuCG_TEST.cu CuCG.cu my_crs_matrix.c


all:
	$(CC) $(CFLAGS) -o CuCG $(SOURCE) $(LDFLAGS)
