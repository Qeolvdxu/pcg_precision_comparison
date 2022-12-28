#include "my_crs_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// find the dot product of two vectors
PRECI_DT dot(PRECI_DT *v, PRECI_DT *u, int n) {

  PRECI_DT x;
int i;

for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;

}


void matvec(my_crs_matrix *A, PRECI_DT* x, PRECI_DT* y)
{
  int n = A->n;

  for (int i = 0; i < n; i++)
    y[i] = 0.0;

  for (int i = 0; i < n; i ++)
    {
      for (int j = A->rowptr[i]; j < A->rowptr[i+1]; j++)
	{
	  y[i] += A->val[j] * x[A->col[j]];
	}
    }
}

int cmp_degree(const void* a, const void* b)
{
  return ( *(int*)a - *(int*)b );
}

my_crs_matrix* rcm_reorder(my_crs_matrix *A)
{
  int n = A->n;
  int *perm = malloc(sizeof(int) * n);
  
  for (int i = 0; i < n; i++)
    perm[i] = i;

  int* degree = malloc(n* sizeof(int));
  for (int i = 0; i < n; i++)
    degree[i] = A->rowptr[i+1] - A->rowptr[i];

  int* sorted = malloc(n * sizeof(int));
  for (int i = 0; i < n; i++)
    sorted[i] = i;

  qsort(sorted, n, sizeof(int), cmp_degree);

  for (int i = 0; i < A->n; i++)
    {
      int root = sorted[i];
      if (perm[root] != root)
	continue;
      perm[root] = perm[perm[root]];
      for (int j = A->rowptr[root]; j < A->rowptr[root+1]; j++)
	{
	  int node = A->col[j];
	  perm[node] = perm[perm[node]];
	}
    }

  my_crs_matrix* B = eye(n);

  // Reorder the rows and columns of the matrix
  int* row_count = calloc(A->n, sizeof(int));
  for (int i = 0; i < A->n; i++) {
    int row = perm[i];
    row_count[row]++;
  }
  for (int i = 0; i < A->n; i++) {
    B->rowptr[i + 1] = B->rowptr[i] + row_count[i];
  }
  for (int i = 0; i < A->n; i++) {
    int row = perm[i];
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++) {
      int col = A->col[j];
      int index = B->rowptr[row] + row_count[row] - 1;
      B->col[index] = col;
      B->val[index] = A->val[j];
      row_count[row]--;
    }
  }

  // Free allocated memory
  /*free(perm);
    free(degree);
  free(sorted);
  free(row_count);*/

  return B;
}

void precondition(my_crs_matrix* M, PRECI_DT* x, PRECI_DT* y)
{
  int n = M->n;
  PRECI_DT sum = 0.0;


  for (int i = 0; i < n; i++)
    {
      y[i] = 0.0;
    }

  //solve M * y = x using foward and back substitution
  for (int i = 0; i < n; i++)
    {
      sum = 0.0;
      for (int j = M->rowptr[i]; j < i; j++)
	sum += M->val[j] * y[M->col[j]];
      y[i] = (x[i] - sum) / M->val[M->rowptr[i]];
    }

  for (int i = n-1; i >= 0; i--)
    {
      sum = 0.0;
      for (int j = i+1; j < M->rowptr[i+1]; j++)
	sum += M->val[j] * y[M->col[j]];
      y[i] = (y[i] - sum) / M->val[M->rowptr[i]];
    }


}

PRECI_DT matvec_dot(my_crs_matrix *A, PRECI_DT* x, PRECI_DT* y, int n)
{
  PRECI_DT result = 0.0;
  for (int i = 0; i < n; i++)
    {
      for (int j = A->rowptr[i]; j < A->rowptr[i+1]; j++)
	{
	  result += x[i] * A->val[j] * y[A->col[j]];
	  // printf("result += %lf * %lf * %lf\n", x[i] , A->val[j] , y[A->col[j]]);
	}
      /*     if (result != result && i % 20 == 0)
	     printf("NaN moment :(\n");*/



    }
  return result;
}

// find the norm of a vector
PRECI_DT norm(int n, PRECI_DT *v) {

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
    fscanf(file, PRECI_S, &M->val[i]);

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
