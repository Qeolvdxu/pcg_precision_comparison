# PCG Precision Comparisons

Compare data types to see if the floating point precision will change the number of iterations and answer vectors of a preconditioned conjugate gradient algorithm across the CPU and GPU.

## Dependencys
gcc/clang, nvcc, cuda capable gpu

## How to use
* Pick all the matrix market matricies (.MTX files) to test and put them in the test_subjects directory
* build the program with makefile
* The output will write to a csv in real time
