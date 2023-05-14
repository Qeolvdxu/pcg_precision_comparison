#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/my_crs_matrix.h"

my_crs_matrix *sparse_transpose(my_crs_matrix *input) {

  my_crs_matrix *res = malloc(sizeof(my_crs_matrix));
  int i;

  res->m = input->m;
  res->n = input->n;
  res->nz = input->nz;

  res->val = malloc(sizeof(PRECI_DT) * res->nz);

  res->col = malloc(sizeof(int) * res->nz);
  res->rowptr = malloc(sizeof(int) * (res->n + 2));

  for (i = 0; i < res->n + 2; i++)
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
  for (int i = 2; i < res->n + 2; ++i) {
    // create incremental sum
    res->rowptr[i] += res->rowptr[i - 1];
  }

  // perform the main part
  for (int i = 0; i < input->n; ++i) {
    for (int j = input->rowptr[i]; j < input->rowptr[i + 1]; ++j) {
      // calculate index to transposed matrix at which we should place current
      // element, and at the same time build final rowPtr
      const int new_index = res->rowptr[input->col[j] + 1]++;
      res->val[new_index] = input->val[j];
      res->col[new_index] = i;
    }
  }
  // res->rowptr = realloc(res->rowptr,res->n*sizeof(PRECI_DT));; // pop that
  // one extra

  return res;
}

// read matrix file into a my_csr_matrix variable
my_crs_matrix *my_crs_read(char *name) {
  my_crs_matrix *M = malloc(sizeof(my_crs_matrix));
  FILE *file;
  if ((file = fopen(name, "r"))) {
    int i;

    fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
    M->val = malloc(sizeof(PRECI_DT) * M->nz);

    M->col = malloc(sizeof(int) * M->nz);
    M->rowptr = malloc(sizeof(int) * (M->n + 1));

    for (i = 0; i <= M->n; i++)
      fscanf(file, "%d ", &M->rowptr[i]);
    for (i = 0; i < M->nz; i++)
      fscanf(file, "%d ", &M->col[i]);
    for (i = 0; i < M->nz; i++)
      fscanf(file, PRECI_S, &M->val[i]);

    /*  printf("CCG rowptr: ");
      for (i = 0; i <= M->n; i++)
        printf("%d ", M->rowptr[i]);
      printf("\n\n");
      printf("CCG col: ");
      for (i = 0; i < M->nz; i++)
        printf("%d ", M->col[i]);
      printf("\n\n");
      printf("CCG val: ");
      for (i = 0; i < M->nz; i++)
        printf(PRECI_S, M->val[i]);
      printf("\n\n");
  */

    fclose(file);
  } else {
    printf("ERROR: could not open file %s\n", name);
    M->n = -1;
  }
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

  for (i = 0; i < M->n - 1; i++)
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
  int n = M->n;
  int m = M->m;
  int nz = M->nz;
  printf("%d %d %d\n", n, m, nz);
  for (i = 0; i < n; i++)
    printf("%d ", M->rowptr[i]);
  printf("\n");

  printf("index,");
  for (i = 0; i < nz; i++)
    printf("%d ", M->col[i]);
  printf("\n");

  printf("values,");
  for (i = 0; i < nz; i++) {
    printf(PRECI_S, M->val[i]);
    printf(" ");
  }
  printf("\n");
}
