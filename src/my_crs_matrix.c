#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/my_crs_matrix.h"

my_crs_matrix *sparse_transpose(my_crs_matrix *input) {
  int n = input->m;
  int m = input->n;
  int nz = input->nz;
  double *val = malloc(nz * sizeof(double));
  int *col = malloc(nz * sizeof(int));
  int *rowptr = malloc((m + 2) * sizeof(int)); // 1 extra

  // Initialize rowptr with zeros
  for (int i = 0; i < m + 2; ++i) {
    rowptr[i] = 0;
  }

  // Count per column
  for (int i = 0; i < nz; ++i) {
    ++rowptr[input->col[i] + 2];
  }

  // From count per column, generate new rowptr (but shifted)
  for (int i = 2; i < m + 2; ++i) {
    // Create incremental sum
    rowptr[i] += rowptr[i - 1];
  }

  // Perform the main part
  for (int i = 0; i < n; ++i) {
    for (int j = input->rowptr[i]; j < input->rowptr[i + 1]; ++j) {
      // Calculate index to transposed matrix at which we should place the
      // current element, and at the same time build the final rowptr
      const int new_index = rowptr[input->col[j] + 1]++;
      val[new_index] = input->val[j];
      col[new_index] = i;
    }
  }

  // Remove the extra element from rowptr
  rowptr[m + 1] = 0;

  my_crs_matrix *result = malloc(sizeof(my_crs_matrix));
  result->m = m;
  result->n = n;
  result->nz = nz;
  result->val = val;
  result->col = col;
  result->rowptr = rowptr;

  return result;
}

// read matrix file into a my_csr_matrix variable
my_crs_matrix *my_crs_read(char *name) {
  my_crs_matrix *M = malloc(sizeof(my_crs_matrix));
  if (M == NULL) {
    printf("ERROR: Memory allocation failed for my_crs_matrix.\n");
    return NULL;
  }

  FILE *file = fopen(name, "r");
  if (file == NULL) {
    printf("ERROR: Failed to open file %s.\n", name);
    free(M);
    return NULL;
  }

  if (fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz) != 3) {
    printf("ERROR: Invalid file format.\n");
    fclose(file);
    free(M);
    return NULL;
  }

  M->val = malloc(sizeof(double) * M->nz);
  if (M->val == NULL) {
    printf("ERROR: Memory allocation failed for val array.\n");
    fclose(file);
    free(M);
    return NULL;
  }

  M->col = malloc(sizeof(int) * M->nz);
  if (M->col == NULL) {
    printf("ERROR: Memory allocation failed for col array.\n");
    fclose(file);
    free(M->val);
    free(M);
    return NULL;
  }

  M->rowptr = malloc(sizeof(int) * (M->n + 1));
  if (M->rowptr == NULL) {
    printf("ERROR: Memory allocation failed for rowptr array.\n");
    fclose(file);
    free(M->col);
    free(M->val);
    free(M);
    return NULL;
  }

  int i;
  for (i = 0; i <= M->n; i++) {
    if (fscanf(file, "%d ", &M->rowptr[i]) != 1) {
      printf("ERROR: Invalid file format.\n");
      fclose(file);
      free(M->rowptr);
      free(M->col);
      free(M->val);
      free(M);
      return NULL;
    }
  }

  for (i = 0; i < M->nz; i++) {
    if (fscanf(file, "%d ", &M->col[i]) != 1) {
      printf("ERROR: Invalid file format.\n");
      fclose(file);
      free(M->rowptr);
      free(M->col);
      free(M->val);
      free(M);
      return NULL;
    }
  }

  for (i = 0; i < M->nz; i++) {
    if (fscanf(file, "%lf ", &M->val[i]) != 1) {
      printf("ERROR: Invalid file format.\n");
      fclose(file);
      free(M->rowptr);
      free(M->col);
      free(M->val);
      free(M);
      return NULL;
    }
  }

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

  M->val = malloc(sizeof(double) * M->nz);
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
    printf("%lf ", M->val[i]);
    printf(" ");
  }
  printf("\n");
}
