#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/trisolv.h"

void forwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x) {
  int n = A->n;
  double sum = 0.0;
  int row_start, row_end, j;
  for (int i = 0; i < n; i++) {
    row_start = A->rowptr[i];
    row_end = A->rowptr[i + 1];
    sum = 0.0;
    for (j = row_start; j < row_end - 1; j++) {
      int col = A->col[j];
      sum += A->val[j] * x[col];
    }
    x[i] = (b[i] - sum) / A->val[row_end - 1];
#ifdef ENABLE_TESTS
    // Print intermediate results
    // printf("FORSUB Iteration %d:\n", i);
    // printf("answer[%d] = (%f - %f) / %f = %f\n", i, b[i], sum,
    //       A->val[row_end - 1], x[i]);
#endif
  }
}

void backwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x) {
  int n = A->n;
  double sum = 0.0;
  int row_start, row_end, j;
  for (int i = n - 1; i >= 0; --i) {
    row_start = A->rowptr[i];
    row_end = A->rowptr[i + 1];
    sum = 0.0;
    for (j = row_end - 1; j > row_start; --j) {
      int col = A->col[j];
      sum += A->val[j] * x[col];
    }
    x[i] = (b[i] - sum) / A->val[row_start];

#ifdef ENABLE_TESTS
    // Print intermediate results
    // printf("BACKSUB Iteration %d:\n", i);
    // printf("answer[%d] = (%f - %f) / %f = %f\n", i, b[i], sum,
    //      A->val[row_start], x[i]);
#endif
  }
}

// s_abft L x y = r for y
int s_abft_forsub(double *val, int *col, int *rowptr, int n, double *r,
                  double *y, double tol) {
  // printVector("vector 1", r, n);
  // printVector("vector 2", y, n);
  //  Compute triangular matrix col checksum
  double *p = (double *)calloc(n, sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      int col_idx = col[j];
      p[col_idx] += val[j];
      // printf("forsub iter %d,%d : \n", j, i);
      // printf(" rowptr :%d\n", rowptr[i]);
      // printf(" col_idx :%d\n", col_idx);
      // printf(" %11lf += %11lf\n\n", p[col_idx], val[j]);
    }
  }

  double dot_product = 0; // inner product between p^T and y
  double S = 0;           // Checksum of vector r
  for (int i = 0; i < n; i++) {
    dot_product += p[i] * y[i];
    // printf("dp %11lf += %11lf * %11lf\n", dot_product, p[i], y[i]);
    S += r[i];
    // printf("S : %11lf\n", S);
  }
  free(p);

#ifdef ENABLE_TESTS
  printf("dp(%lf) == S(%lf)\n", dot_product, S);
#endif

  double diff = fabs(dot_product - S);
  if (diff <= tol)
    return 0;
  else
    return 1;
}

// s_abft U x y = r for y
int s_abft_backsub(double *val, int *col, int *rowptr, int n, double *r,
                   double *y, double tol) {
  // printVector("vector 1", r, n);
  // printVector("vector 2", y, n);
  //  Compute triangular matrix col checksum
  double *p = (double *)calloc(n, sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      // printf("backsub iteration %d,%d : \n", j, i);
      p[i] += val[j];
      // printf(" %11lf += %11lf\n\n", p[i], val[j]);
    }
  }

  double dot_product = 0; // inner product between p^T and y
  double S = 0;           // Checksum of vector r
  for (int i = 0; i < n; i++) {
    dot_product += p[i] * y[i];
    S += r[i];
  }
  free(p);

#ifdef ENABLE_TESTS
  printf("dp(%lf) == S(%lf)\n", dot_product, S);
#endif

  double diff = fabs(dot_product - S);
  if (diff <= tol)
    return 0;
  else
    return 1;
}

// s_abft A * b = c for c
int s_abft_spmv(double *val, int *col, int *rowptr, int n, double *b, double *c,
                double tol) {
  // Compute A col checksum
  double *p = (double *)calloc(n, sizeof(double));
  for (int j = 0; j < n; j++) {
    for (int i = rowptr[j]; i < rowptr[j + 1]; i++) {
      p[j] += val[i];
    }
  }

  double dot_product = 0; // inner product between p^T and y
  double S = 0;           // Checksum of vector r
  for (int i = 0; i < n; i++) {
    dot_product += p[i] * b[i];
    S += c[i];
  }
  free(p);

#ifdef ENABLE_TESTS
  printf("dp(%.10lf) == S(%.10lf)\n", dot_product, S);
#endif

  double diff = fabs(dot_product - S);
  if (diff <= tol)
    return 0;
  else
    return 1;
}

int isLowerTriangular(my_crs_matrix *A) {
  for (int i = 0; i < A->n; i++) {
    int row_start = A->rowptr[i];
    int row_end = A->rowptr[i + 1];
    for (int j = row_start; j < row_end; j++) {
      int col = A->col[j];
      if (col > i) {
        return 0; // Found an element above the diagonal
      }
    }
  }
  return 1; // Matrix is lower triangular
}

void printVector(const char *name, double *vec, int size) {
  printf("%s:\n", name);
  long double sum = 0.0;
  for (int i = 0; i < size; i++) {
    printf("%.10f, ", vec[i]);
    sum += vec[i];
  }
  printf("\n%Lf\n\n", sum);
}

void printMatrix(const char *name, my_crs_matrix *mat) {
  printf("%s:\n", name);
  long double sum = 0.0;
  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->n; j++) {
      int found = 0;
      int row_start = mat->rowptr[i];
      int row_end = mat->rowptr[i + 1];
      for (int k = row_start; k < row_end; k++) {
        if (mat->col[k] == j) {
          printf("%.10f, ", mat->val[k]);
          sum += mat->val[k];
          found = 1;
          break;
        }
      }
      if (!found) {
        printf("0.00, ");
      }
    }
    printf("\n");
  }
  printf("\n%Lf\n\n", sum);
}

int checkSolution(my_crs_matrix *A, double *b, double *x, double tolerance) {
  int n = A->n;
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    int row_start = A->rowptr[i];
    int row_end = A->rowptr[i + 1];
    for (int j = row_start; j < row_end; j++) {
      int col = A->col[j];
      sum += A->val[j] * x[col];
    }
    if (fabs(sum - b[i]) > tolerance) {
      return 0; // Solution is incorrect
    }
  }
  return 1; // Solution is correct
}

/*int main() {
  // Create and initialize the my_crs_matrix struct
  my_crs_matrix *A = my_crs_read(
      "../test_subjects/precond_norm/bcsstk08.mtx.PRECONDITIONER.mtx.csr");
  int n = A->n;

  if (A->val == NULL)
    return 1;

  if (!isLowerTriangular(A)) {
    printf("Error: The matrix is not lower triangular.\n");
    // Free memory
    free(A->val);
    free(A->col);
    free(A->rowptr);
    return 1;
  }

  double *b = malloc(sizeof(double) * n); // Right-hand side vector
  double *x = malloc(sizeof(double) * n); // Solution vector

  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    b[i] = (double)(rand() % 10000) / 100.0 + 1.0;
    x[i] = 0;
  }

  printMatrix("Matrix A", A);
  printVector("Vector b", b, n);

  forwardSubstitutionCSR(A, b, x);

  printVector("Solution x", x, n);

  // Check the solution
  double tolerance = 1e-6; // Tolerance for solution comparison
  int isCorrect = checkSolution(A, b, x, tolerance);

  if (isCorrect) {
    printf("Solution is correct.\n");
  } else {
    printf("Solution is incorrect.\n");
  }

  // Free memory
  free(A->val);
  free(A->col);
  free(A->rowptr);

  return 0;
}*/
