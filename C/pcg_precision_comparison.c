#include <stdio.h>
#include <math.h>

typedef struct
{
  int values;
  int columns;
  int rowptr;
} my_csr_matrix;


static double norm(int n, double*v)
{
  double norm;
  int i;

  for (i=0, norm=0.0; i<n; i++) norm += v[i]*v[i];

  norm = sqrt(norm);
  return norm;
}


int my_pcg(my_csr_matrix* A,
	   int* b,
	   int maxiter,
	   double tol,
	   my_csr_matrix* M)
{
  // Set up
  //i = 0
  int i,j,k;
  int Iter;
  int n;  
  //MT = M'
  int* MT = malloc(1);
  /*for (int i = 0; i < A_size; ++i) {
    for (int j = 0; j < A_size; ++j) {
    MT[j][i] = A[i][j];
    }
    }*/

  //x = zeros(size(A,1), 1);
  double* x = malloc(1);
  double inorm;
  
  //r = b - A*x

  //z = MT(M\r)

  // p = x

  n = A->columns;
  double* R = malloc(1);
  inorm = norm(n,R);
  
  // Iterate
  Iter = 0;
  while( i <= maxiter && norm(n,R)/inorm > tol)
    {
      Iter++;
    }
}

int main(int argc, char *argv[])
{
  //my_pcg();
  return 0;
}

