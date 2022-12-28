#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "my_crs_matrix.h"

void conjugant_gradient(my_crs_matrix *A, my_crs_matrix *M, PRECI_DT* b, PRECI_DT *x, int max_iter, PRECI_DT tolerance)
{
  int n = A->n;
  PRECI_DT* r = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* p = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* q = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* z = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));

  matvec(A, x, r);

  for (int i = 0; i < n; i++)
  {
    r[i] = b[i] - r[i];
    p[i] = r[i];
  }

  PRECI_DT r_norm_squared = dot(r,r,n);
  PRECI_DT b_norm_squared = dot(b,b,n);

  PRECI_DT r_new_norm_squared = dot(r,r,n);
  PRECI_DT beta = r_new_norm_squared / r_norm_squared;

  PRECI_DT Ap_dot_p = matvec_dot(A, p, p, n);
  PRECI_DT p_dot_z = dot(p, z, n);
  PRECI_DT alpha = r_norm_squared / p_dot_z;

  PRECI_DT r_norm = sqrt(r_norm_squared);
  for (int iter = 0; iter < max_iter; iter++)
    {
      // apply precondition
      precondition(M,p,z);

      // find alpha
      Ap_dot_p = matvec_dot(A, p, p, n);
      p_dot_z = dot(p, z, n);
      alpha = r_norm_squared / p_dot_z;


      // Update residual and solution
      for (int i = 0; i < n; i++)
	{
	  x[i] += alpha * p[i];
	  r[i] -= alpha * Ap_dot_p;
	}

      printf("iteration %d\n x0 = %lf \t alpha= %lf \t beta= %lf \n r0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n\n\n",iter, x[0], alpha, beta,r[0],p[0],q[0],z[0]);


      // Check if converged with given tolerance
      r_norm = sqrt(r_norm_squared);
      if (r_norm < tolerance * sqrt(b_norm_squared))
	break;

      // find beta
      r_new_norm_squared = dot(r,r,n);
      beta = r_new_norm_squared / r_norm_squared;


      // Update search direction
      for (int i = 0; i < n; i++)
	p[i] = r[i] + beta * p[i];

      r_norm_squared = r_new_norm_squared;
    }
  free(r);
  free(p);
  free(q);
  free(z);
}

int main(int argc, char* argv[]) {
  int i, tests;
  int test_count = 1;
  my_crs_matrix *test;
  my_crs_matrix *test_RCM;
  my_crs_matrix *precond;

  if (argc != 3)
    {
      printf("ERROR: command line arguments invalid/missing\n");
      return 1;
    }
  char *files[18] = {
    "./test_subjects/494_bus.mtx.crs",
    "./test_subjects/662_bus.mtx.crs",
    "./test_subjects/685_bus.mtx.crs",
    "./test_subjects/dwt_869.mtx.crs",
    "./test_subjects/bcsstk01.mtx.crs",
"./test_subjects/bcsstk02.mtx.crs",
      "./test_subjects/bcsstk03.mtx.crs",
      "./test_subjects/bcsstk04.mtx.crs",
      "./test_subjects/bcsstk05.mtx.crs",
      "./test_subjects/bcsstk06.mtx.crs",
      "./test_subjects/bcsstk07.mtx.crs",
      "./test_subjects/bcsstk08.mtx.crs",
      "./test_subjects/bcsstk09.mtx.crs",
      "./test_subjects/bcsstk10.mtx.crs",
      "./test_subjects/bcsstk11.mtx.crs",
      "./test_subjects/bcsstk12.mtx.crs",
      "./test_subjects/bcsstk13.mtx.crs",
      "./test_subjects/bcsstk14.mtx.crs"
    };
    PRECI_DT tol = (float)atof(argv[2]);

    FILE *ofile = fopen("C_cg-results.csv","w");
    int iter, maxit;
    for (tests=0; tests<test_count;tests++)
      {
	printf("\n%s\n",files[tests]);

	maxit = atoi(argv[1])-1;
	test = my_crs_read(files[tests]);//"./test_subjects/bcsstk10.mtx.crs");
	test_RCM = rcm_reorder(test);
	precond = eye(test->n);
	PRECI_DT* b;
	PRECI_DT* x;

	b = malloc(sizeof(PRECI_DT) * test->n);
	x = calloc(test->n, sizeof(PRECI_DT));

	// b vector of 1s
	for (i = 0; i < test->n; i++)
			       b[i] = 1;

			     // apply CG
			     printf("calling cg\n");
			     conjugant_gradient(test_RCM, precond, b, x, maxit, tol);



			     free(b);
			     free(x);

			     fprintf(ofile,"%s,",files[tests]);
			     fprintf(ofile,"%d,",iter);
			     for (i = 0; i < test->n; i++)
			       fprintf(ofile,"%f,",x[i]);
	fprintf(ofile,"\n");


      }

    printf(" donee \n");

    my_crs_free(test);
    return 0;
  }
