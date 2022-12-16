all:
	make clean
	make C_cg
	make CUDA_cg

C_cg:
	make -f C_cg.mk

CUDA_cg:
	make -f CUDA_cg.mk

clean:
	rm C_cg
	rm CUDA_cg
