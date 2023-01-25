all:
	nvcc -c -o CuCG.o CuCG.cu
	clang -o cgpc -lcuda -lcudart main.c CuCG.o

CCG_TEST_M:
	make -f CCG_TEST.mk

CUDA_cg_M:
	make -f CuCG_TEST.mk

clean:
	rm CCG_TEST
	rm CuCG_TEST
