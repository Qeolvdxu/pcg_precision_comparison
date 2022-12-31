all:
	make CCG_TEST
	make CUDA_cg

CCG_TEST:
	make -f CCG_TEST.mk

CUDA_cg:
	make -f CUDA_cg.mk

clean:
	rm C_cg
	rm CUDA_cg
