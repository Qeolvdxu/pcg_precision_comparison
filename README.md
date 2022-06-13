# pcg_bfloat_comparison

Octave/Matlab program to compare data types to see if the floating level will change the interaction counts when running a precondition conjugate gradient algorithm. 

## Requires
This program requires :
* Chop : https://github.com/higham/chop
* MMRead : https://www.mathworks.com/matlabcentral/fileexchange/8028-mmread

## How to use
* Pick all the matricies to test and put them in the test_subjects directory
* run pcg_bfloat_comparison.m with Matlab or Octave with the job file or on its own
* The output will print in real time but also generate a results.cvs file
