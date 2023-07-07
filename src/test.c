#include <stdio.h>
#include <stdlib.h>

typedef struct {
  double *data;
  int *indices;
  int *indptr;
  int nrows;
} CSRMatrix;

void forward_substitution(CSRMatrix *csr_matrix, double *b, double *x) {
  int n = csr_matrix->nrows;

  for (int i = 0; i < n; i++) {
    int start = csr_matrix->indptr[i];
    int end = csr_matrix->indptr[i + 1];
    double *row_vals = &csr_matrix->data[start];
    int *row_indices = &csr_matrix->indices[start];

    x[i] = b[i];
    for (int j = start; j < end; j++) {
      int col_index = row_indices[j];
      double val = row_vals[j];
      if (col_index < i) {
        x[i] -= val * x[col_index];
      }
    }
    x[i] /= csr_matrix->data[start];
  }
}

int main() {
  double data[] = {1, 2, 3, 4, 5, 6};
  int indices[] = {0, 0, 1, 0, 1, 2};
  int indptr[] = {0, 2, 5, 6};
  int nrows = 3;
  double b[] = {1, 2, 3};
  double x[nrows];

  CSRMatrix csr_matrix;
  csr_matrix.data = data;
  csr_matrix.indices = indices;
  csr_matrix.indptr = indptr;
  csr_matrix.nrows = nrows;

  forward_substitution(&csr_matrix, b, x);

  printf("Solution:\n");
  for (int i = 0; i < nrows; i++) {
    printf("%f\n", x[i]);
  }

  return 0;
}
