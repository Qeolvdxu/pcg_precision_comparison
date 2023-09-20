#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/trisolv.h"

void vecErrorInj(double *p, int vector_size, int k)
{

  // Generate a random error e in the range of values in vector p
  double min_value = p[0];
  double max_value = p[0];
  for (int i = 1; i < vector_size; i++)
  {
    if (p[i] < min_value)
      min_value = p[i];
    if (p[i] > max_value)
      max_value = p[i];
  }
  double e = ((double)rand() / RAND_MAX) * (max_value - min_value);
  // printf("MIN = %lf     MAX = %lf\n",min_value,max_value);

  // Inject the error into vector p
  p[k] += e;
  printf("CPU ERROR INJECTED INTO p[%d] += %lf = %lf\n", k, e, p[k]);
}

void vecErrorInj_gpu(double *p, int vector_size, int k)
{
  // Copy the vector from GPU to CPU
  double *h_p = (double *)malloc(sizeof(double) * vector_size);
  cudaMemcpy(h_p, p, sizeof(double) * vector_size, cudaMemcpyDeviceToHost);

  // Generate a random error e in the range of values in the vector
  double min_value = h_p[0];
  double max_value = h_p[0];
  for (int i = 1; i < vector_size; i++)
  {
    if (h_p[i] < min_value)
      min_value = h_p[i];
    if (h_p[i] > max_value)
      max_value = h_p[i];
  }
  double e = ((double)rand() / RAND_MAX) * (max_value - min_value);

  // Inject the error into the vector
  h_p[k] += e;

  // Copy the modified vector back to the GPU
  cudaMemcpy(p, h_p, sizeof(double) * vector_size, cudaMemcpyHostToDevice);

  // Free memory on the CPU
  free(h_p);
  // printf("GPU ERROR INJECTED INTO p[%d] += %lf = %lf\n", k, e, h_p[k]);
}

double sp2nrmrow(int row_number, int num_rows, int *rowptr, double *val)
{
  double sum_squared_elements = 0.0;

  // Loop through the elements of the given row
  for (int j = rowptr[row_number]; j < rowptr[row_number + 1]; j++)
  {
    sum_squared_elements += val[j] * val[j];
  }

  return sqrt(sum_squared_elements);
}

void forwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x)
{
  int n = A->n;
  double sum = 0.0;
  int row_start, row_end, j;
  for (int i = 0; i < n; i++)
  {
    row_start = A->rowptr[i];
    row_end = A->rowptr[i + 1];
    sum = 0.0;
    for (j = row_start; j < row_end - 1; j++)
    {
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

void backwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x)
{
  int n = A->n;
  double sum = 0.0;
  int row_start, row_end, j;
  for (int i = n - 1; i >= 0; --i)
  {
    row_start = A->rowptr[i];
    row_end = A->rowptr[i + 1];
    sum = 0.0;
    for (j = row_end - 1; j > row_start; --j)
    {
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
                  double *y, double tol)
{
  // printVector("vector 1", r, n);
  // printVector("vector 2", y, n);
  //  Compute triangular matrix col checksum
  double *p = (double *)calloc(n, sizeof(double));
  for (int i = 0; i < n; i++)
  {
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
    {
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
  for (int i = 0; i < n; i++)
  {
    dot_product += p[i] * y[i];
    // printf("dp %11lf += %11lf * %11lf\n", dot_product, p[i], y[i]);
    S += r[i];
    // printf("S : %11lf\n", S);
  }
  free(p);

#ifdef ENABLE_TESTS
#endif
  double diff = fabs(dot_product - S) * tol;
  /// printf("dp(%.10lf) == S(%.10lf)\n", dot_product, S);
  // printf("%.10lf <= %.10lf\n", diff, tol);
  if (diff <= tol)
    return 0;
  else
    return 1;
}
// s_abft U x y = r for y
int s_abft_backsub(double *val, int *col, int *rowptr, int n, double *r,
                   double *y, double tol)
{
  // printVector("vector 1", r, n);
  // printVector("vector 2", y, n);
  //   Compute triangular matrix col checksum
  double *p = (double *)calloc(n, sizeof(double));
  for (int i = 0; i < n; i++)
  {
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
    {
      //     printf("backsub iteration %d,%d : \n", j, i);
      p[i] += val[j];
      //     printf(" %11lf += %11lf\n\n", p[i], val[j]);
    }
  }

  double dot_product = 0; // inner product between p^T and y
  double S = 0;           // Checksum of vector r
  for (int i = 0; i < n; i++)
  {
    dot_product += p[i] * y[i];
    S += r[i];
  }
  free(p);

#ifdef ENABLE_TESTS
  printf("dp(%lf) == S(%lf)\n", dot_product, S);
#endif

  double diff = fabs(dot_product - S) * tol;
  // printf("dp(%.10lf) == S(%.10lf)\n", dot_product, S);
  // printf("%.10lf <= %.10lf\n", diff, tol);
  if (diff <= tol)
    return 0;
  else
    return 1;
}

int selective_abft()
{
  return 0;
}

void calculate_checksum(double *val, int *col, int *rowptr, int n, double *checksum)
{
  int i, j;

  for (j = 0; j < n; j++)
  {
    for (i = rowptr[j]; i < rowptr[j + 1]; i++)
    {
      checksum[col[i]] += val[i];
    }
  }
}

// Full classical SPMV fault checking, 100% overhead
// t = A * p
int abft_spmv_selective(double *val, int *col, int *rowptr, int n, double *p, double *t, double *buff, double tol, int k, int *critindex)
{
  int col_index, row_index;

  // SPMV
  for (int i = 0; i < k; i++) {
      buff[critindex[i]] = 0;
      col_index = critindex[i];
      for (int j = rowptr[col_index]; j < rowptr[col_index + 1]; j++) {
          row_index = col[j];
          buff[row_index] += val[j] * p[col_index];
      }
  }

  // Compare vectors y and t
  for (int i = 0; i < n; i++)
  {
    printf("abft: %.10lf vs\n%.10lf\n",buff[i],t[i]);
    if (fabs(buff[i] - t[i]) > tol)
      return 1;
  }

  return 0;
}

// s_abft t = A * p
int s_abft_spmv(double *acChecksum, int n, double *p, double *t,
                double tol)
{
  int i;
  long double *dotp = calloc(n, sizeof(long double));
  //  printf("TVECONE = %.50lf\n", t[1]);

  // Calculate acChecksum for each column of the matrix A

  for (i = 0; i < n; i++)
  {
    dotp[i] = acChecksum[i] * p[i];
    // printf("%Lf = %Lf * %lf\n", dotp[i], acChecksum[i], p[i]);
  }

  long double tcSum = 0.0;
  long double dotpSum = 0.0;

  for (i = 0; i < n; i++)
  {
    tcSum += t[i];
    dotpSum += dotp[i];
  }
  tcSum = tcSum * tol;
  dotpSum = dotpSum;

  // free(acChecksum);
  free(dotp);

  // printf("tol: %lf\n", tol);
  //  Check conditions (i) and (ii) of Theorem 1
  // printf("1: %.50Lf == %.50Lf\n", tcSum, dotpSum);

  printf("2: %.50lf <= %.50lf\n", fabs(tcSum - dotpSum), tol);
  if (fabs(tcSum - t[n]) <= tol && fabs(tcSum - dotpSum) <= tol)
  {
    // if (fabs(tcSum - dotpSum) <= tol) {
    //    printf("winner");
    return 0; // SpMV operation is correct based on Theorem 1
  }
  else
  {
    //   printf("loser");
    return 1; // Error occurred during the SpMV operation
  }
}

int isLowerTriangular(my_crs_matrix *A)
{
  for (int i = 0; i < A->n; i++)
  {
    int row_start = A->rowptr[i];
    int row_end = A->rowptr[i + 1];
    for (int j = row_start; j < row_end; j++)
    {
      int col = A->col[j];
      if (col > i)
      {
        return 0; // Found an element above the diagonal
      }
    }
  }
  return 1; // Matrix is lower triangular
}

void printVector(const char *name, double *vec, int size)
{
  printf("%s:\n", name);
  long double sum = 0.0;
  for (int i = 0; i < size; i++)
  {
    printf("%.10f, ", vec[i]);
    sum += vec[i];
  }
  printf("\n%Lf\n\n", sum);
}

void printMatrix(const char *name, my_crs_matrix *mat)
{
  printf("%s:\n", name);
  long double sum = 0.0;
  for (int i = 0; i < mat->n; i++)
  {
    for (int j = 0; j < mat->n; j++)
    {
      int found = 0;
      int row_start = mat->rowptr[i];
      int row_end = mat->rowptr[i + 1];
      for (int k = row_start; k < row_end; k++)
      {
        if (mat->col[k] == j)
        {
          printf("%.10f, ", mat->val[k]);
          sum += mat->val[k];
          found = 1;
          break;
        }
      }
      if (!found)
      {
        printf("0.00, ");
      }
    }
    printf("\n");
  }
  printf("\n%Lf\n\n", sum);
}

int checkSolution(my_crs_matrix *A, double *b, double *x, double tolerance)
{
  int n = A->n;
  for (int i = 0; i < n; i++)
  {
    double sum = 0.0;
    int row_start = A->rowptr[i];
    int row_end = A->rowptr[i + 1];
    for (int j = row_start; j < row_end; j++)
    {
      int col = A->col[j];
      sum += A->val[j] * x[col];
    }
    if (fabs(sum - b[i]) > tolerance)
    {
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
