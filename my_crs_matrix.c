#include "my_crs_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// find the dot product of two vectors
static PRECI_DT dotprod(int n, PRECI_DT *v, PRECI_DT *u) {
  PRECI_DT x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;

}

// find the norm of a vector
static PRECI_DT norm(int n, PRECI_DT *v) {

  PRECI_DT ssq, scale, absvi;
  int i;

  if (n==1) return fabs(v[0]);

  scale = 0.0;
  ssq = 1.0;

  for (i=0; i<n; i++)
    {
      if (v[i] != 0)
	{
	  absvi = fabs(v[i]);
	  if (scale < absvi)
	    {
	      ssq = 1.0 + ssq * (scale/absvi)*(scale/absvi);
	      scale = absvi;
	    }
	  else
	    ssq = ssq + (absvi/scale)*(absvi/scale);
	}
    }

  return scale * sqrt(ssq);

}

// multiply a my_crs_matrix with a vector
int my_crs_times_vec(my_crs_matrix *M, PRECI_DT *v, PRECI_DT *ans) {
  int i, j;
  for (i = 0; i < M->n; i++) {
    //    printf("test %d/%d\n",i,M->n);
    ans[i] = 0;
    for (j = M->rowptr[i]; j < M->rowptr[i + 1] && j < M->nz; j++)
      {
	/*// DEBUG PRINTS
	  printf("%f : ",ans[i]);

	printf("%f :",M->val[j]);

	printf("%d : ",M->col[j]);

	printf("%f\n",v[M->col[j]]);

	//printf(" |%d| ",j);*/
	ans[i] += M->val[j] * v[M->col[j]];
      }

  }
  return 0;

}
my_crs_matrix *sparse_transpose(my_crs_matrix *input) {

  my_crs_matrix *res = malloc(sizeof(my_crs_matrix));
  int i;

  res->m = input->m;
  res->n = input->n;
  res->nz = input->nz;

  res->val = malloc(sizeof(PRECI_DT) * res->nz);
  res->col = malloc(sizeof(int) *res->nz);
  res->rowptr = malloc(sizeof(int) * (res->n + 2));

  for (i = 0; i < res->n+2; i++)
    res->rowptr[i] = 0;
  for (i = 0; i < res->nz; i++)
    res->col[i] = 0;
  for (i = 0; i < res->nz; i++)
    res->val[i] = 0;


  // count per column
  for (int i = 0; i < input->nz; ++i) {
    ++res->rowptr[input->col[i] + 2];
  }

  // from count per column generate new rowPtr (but shifted)
  for (int i = 2; i < res->n+2; ++i) {
    // create incremental sum
    res->rowptr[i] += res->rowptr[i - 1];
  }

  // perform the main part
  for (int i = 0; i < input->n; ++i) {
    for (int j = input->rowptr[i]; j < input->rowptr[i + 1]; ++j) {
      // calculate index to transposed matrix at which we should place current element, and at the same time build final rowPtr
      const int new_index = res->rowptr[input->col[j] + 1]++;
      res->val[new_index] = input->val[j];
      res->col[new_index] = i;
    }
  }
  //res->rowptr = realloc(res->rowptr,res->n*sizeof(PRECI_DT));; // pop that one extra

  return res;
}

int my_crs_cg(my_crs_matrix *M, PRECI_DT *b, PRECI_DT tol, int maxit, my_crs_matrix *precond, PRECI_DT *x) {
  printf("%d\n",M->col[0]);
  my_crs_matrix *precondTrans = sparse_transpose(precond);
  int i, j;
  int v;
  // allocate vectors
  for (i = 0; i < M->n; i++)
    x[i] = 0;
  PRECI_DT alpha, beta;
  PRECI_DT *p = malloc(sizeof(PRECI_DT) * M->n);
  PRECI_DT *r = malloc(sizeof(PRECI_DT) * M->n);
  PRECI_DT *q = malloc(sizeof(PRECI_DT) * M->n);
  PRECI_DT *z = malloc(sizeof(PRECI_DT) * M->n);
  for (i=0; i<M->n; i++) r[i] = b[i];
  
  PRECI_DT init_norm = norm(M->n,r);
  PRECI_DT norm_ratio = 1;

  // Set up to iterate
  //printf("start cg\n");

  printf("r = ");
  my_crs_times_vec(M, x, r); // ! 
  for (i = 0; i < M->n; i++) {
    r[i] = b[i] - r[i];
    printf("%lf, ", r[i]);
  }
  printf("\n");

  for (i = 0; i < M->n; i++)
    z[i] = r[i];

  for (i = 0; i < M->n; i++)
    p[i] = z[i];

  // Start iteration

  printf("%f > %f ?\n",norm_ratio > tol);
  i = 0;
  while (i <= maxit && norm_ratio > tol) {
    //printf(" ** ITERATION %d ** \n\n",i);
    printf("%d : ",i);

    //printf("x[1] = %f\n",x[1]);
    i++;


    // printf("\n\ni:%d\nnorm_r: %f tol_b: %lf\n", i, norm(M->n, r), norm(M->n, b)*tol);

    //printf("\nnorm_r / norm_b: %f\n", norm(M->n, r) / norm(M->n, b));

    // #update q#
    my_crs_times_vec(M, p, q);

    v = dotprod(M->n, r, z);

    // alpha =  / dot(p,q)
    alpha = v / dotprod(M->n, p, q);
    // products eventually over flow mantisa and turn into inf, then nan after being used

    // x = x + alpha*p
    for (j = 0; j < M->n; j++) {
      x[j] += alpha * p[j];
    }

    // r = r - alpha*q
    printf(" (%f) ",q[1]);
    for (j = 0; j < M->n; j++)
	     r[j] -= alpha * q[j];

    //my_crs_times_vec(M,x,r);
    //for (j = 0; j < M->n; j++)
    // r[j] -= b[j];


  

    for (j = 0; j < M->n; j++)
      z[j] = r[j];

    //printf("beta = %f / %f = ",dotprod(M->n, r, z), v); 
    beta = dotprod(M->n, r, z) / v;
    //printf("%f\n",beta);



    // p = z + beta * p;
    for (j = 0; j < M->n; j++) {
      //printf("p[%d] = %f + %f * %f = ",j,z[j]+beta*p[j]);
      p[j] = z[j] + (beta * p[j]);
      //printf("%f\n",p[j]);

    }
    /* printf("%lf %lf %lf\n", p[j], z[j], beta * p[j]);
       }
    */
    /* // VECTOR PRINTING FOR DEBUG
       printf("\n p vector: ");
       for (j = 0; j < M->n; j++)
       printf("%lf, ", p[j]);
       printf("%lf, ", p[j]);

       printf("\n r vector: ");
       for (j = 0; j < M->n; j++)
       printf("%lf, ", r[j]);
       printf("%lf, ", r[j]);
       printf("\n q vector: ");
       for (j = 0; j < M->n; j++)
       printf("%lf, ", q[j]);
       printf("%lf, ", q[j]);

       printf("\n z vector: ");
       for (j = 0; j < M->n; j++)
       printf("%lf, ", z[j]);
       printf("%lf, ", z[j]);*/

    
    
    //printf("\n alpha: %lf ", alpha);
    //printf("\n beta: %lf ", beta);
    norm_ratio = norm(M->n,r) / init_norm; //norm(M->n,b);//init_norm;
    //if( norm(M->n, r) <= norm(M->n, b) * tol) break;

    printf("%f / %f  >  %f ?\n",init_norm, norm(M->n,r) , tol);


}
  printf("\n *total of %d iterations* \n", i);
  free(p);
  free(q);
  free(z);
  free(r);
  return i;
}
// read matrix file into a my_csr_matrix variable
my_crs_matrix *my_crs_read(char *name) {
  my_crs_matrix *M = malloc(sizeof(my_crs_matrix));
  FILE *file = fopen(name, "r");
  int i;

  fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
  M->val = malloc(sizeof(PRECI_DT) * M->nz);

  M->col = malloc(sizeof(int) * M->nz);
  M->rowptr = malloc(sizeof(int) * M->n);

  for (i = 0; i < M->n; i++)
    fscanf(file, "%d ", &M->rowptr[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, "%d ", &M->col[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, "%f ", &M->val[i]);

  fclose(file);
  return M;
}

// makes identity matrix in csr
my_crs_matrix *eye(int n) {

  my_crs_matrix *M = malloc(sizeof(my_crs_matrix));
  int i;

  M->m = n;
  M->n = n;
  M->nz = n;

  M->val = malloc(sizeof(PRECI_DT) * M->nz);
  M->col = malloc(sizeof(int) * M->nz);
  M->rowptr = malloc(sizeof(int) * M->n);

  for (i = 0; i < M->n; i++)
    M->rowptr[i] = i;
  for (i = 0; i < M->nz; i++)
    M->col[i] = i;
  for (i = 0; i < M->nz; i++)
    M->val[i] = 1;


  return M;
}


// Free my_csr_matrix variable
void my_crs_free(my_crs_matrix *M) {
  free(M->val);
  free(M->col);
  free(M->rowptr);
  free(M);

  return;
}

    // Print my_csr_matrix Matrix
void my_crs_print(my_crs_matrix *M) {
  int i = 0;
  printf("rowptr,");
  for (i = 0; i < M->n; i++)
    printf("%d, ", M->rowptr[i]);
  printf("\n\n");

  printf("index,");
  for (i = 0; i < M->nz; i++)
    printf("%d, ", M->col[i]);
  printf("\n\n");

  printf("values,");
  for (i = 0; i < M->nz; i++) {
    printf(PRECI_S, M->val[i]);
    printf(", ");
  }
  printf("\n\n");
}
