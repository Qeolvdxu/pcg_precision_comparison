#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "my_crs_matrix.h"

int conjugant_gradient(my_crs_matrix *A, my_crs_matrix *M, PRECI_DT* b, PRECI_DT *x, int max_iter, PRECI_DT tolerance)
{
  int n = A->n;
  PRECI_DT* r = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* p = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* q = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));
  PRECI_DT* z = (PRECI_DT*) malloc(n * sizeof(PRECI_DT));

  PRECI_DT alpha = 0.0;
  PRECI_DT beta = 0.0;

  int iter = 0;
  int j = 0;

  PRECI_DT init_norm = norm(n,r);
  PRECI_DT norm_ratio = 1;

  PRECI_DT v = 0;

  // zero fill x
  for (int i = 0; i < n; i++) x[i] = 0;

  for (int i = 0; i < n; i++) p[i] = 1;
  for (int i = 0; i < n; i++) z[i] = 1;

  // P Z AND BETA

  // r = b - A*x
  my_crs_times_vec(A, x, r);
  printf("r[3] = %lf\n",r[3]);
  for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
  printf("r[3] = %lf\n",r[3]);


  printf("iteration PREQUEL\n x0 = %lf \t alpha= %lf \t beta= %lf \n r0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm ratio(%lf) > tolerance(%lf)\n\n\n",x[0], alpha, beta,r[0],p[0],q[0],z[0],norm(n,r)/norm(n,b),tolerance);


	 printf("** %lf | %d | %d ** \n",A->val[1], A->col[1], A->rowptr[1]);
	 // main CG loop
	 while (iter <= max_iter && norm(n,r) / norm(n,b) > tolerance) {
	   // next iteration
	   iter++;



	   // q = A*p
	   my_crs_times_vec(A, p, q);

	   // v = early dot(r,z) 
	   v = dot(r,z,n);

	   //printf("p*q[1]=%lf\n",dot(p,q,n));
	   // alpha = v / dot(p,q)
	   alpha = v / dot(p, q, n);


	   // x = x + alpha * p
	   for (j = 0; j < n; j++)
	     x[j] += alpha * p[j];

	   // r = r - alpha * q
	   for (j = 0; j < n; j++)
	     r[j] -= alpha * q[j];

	   // Precondition
	   precondition(M,r,z);
	   //for (j = 0; j < n; j++) z[j] = 1;


	   // beta = dot(r,z) / v
	   beta = dot(r, z, n) / v;

	   // p = z + beta * p
	   for (j = 0; j < n; j++) 
	     p[j] = z[j] + (beta * p[j]);

	   printf("end of iteration %d\n x0 = %lf \t alpha= %lf \t beta= %lf \n v = %lf\nr0 = %lf \n p0 = %lf\n q0 = %lf\n z0 = %lf\n if (norm ratio(%lf) > tolerance(%lf)\n\n\n",iter, x[1], alpha, beta,v,r[0],p[0],q[0],z[0],norm(n,r)/norm(n,b),tolerance);

	 }
  free(r);
  free(p);
  free(q);
  free(z);
  return iter;
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
	iter = conjugant_gradient(test, precond, b, x, maxit, tol);



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
