# PCG Precision Comparisons

Compare data types to see if the floating point precision will change the number of iterations and answer vectors of a preconditioned conjugate gradient algorithm across the CPU and GPU.

Timings for the GPU and CPU threads are available.

## Dependencys
gcc/clang, nvcc, cuda capable gpu

## How to use
There are two options

Manually:
* Pick all the matrix market matricies (.MTX files) to test and put them in the test_subjects/mm directory
* Run the octave script converter.m in the scripts directory
* build the program with makefile
  * you can use the -dcpu_precision=single/double flag and the -dgpu_precision=single/double flag to set precision levels
  * there is also the -dcpu_mode=debug and -dgpu_mode=debug flags to enable per iteration calculation printing (THIS WILL MESS UP TIMINGS!)  
* The output will write to two different csvs per device

Runner Script:
* Use the runner.sh script in the scripts directory
* This will order matricies and precondtion, build to project, run the project, and combine the two csvs into combo.csv
