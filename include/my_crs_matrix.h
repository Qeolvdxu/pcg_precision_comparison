#ifndef MY_CSR_MATRIX_H_
#define MY_CSR_MATRIX_H_

// compressed sparse row matrix
typedef struct {
  int n;
  int m;
  int nz;
  double *val;
  int *col;
  int *rowptr;
} my_crs_matrix;

my_crs_matrix *sparse_transpose(my_crs_matrix *matrix);

// print each vector
void my_crs_print(my_crs_matrix *M);

// create matrix
my_crs_matrix *my_crs_read(char *name);

// free the matrix from memory
void my_crs_free(my_crs_matrix *M);

#endif // MY_CSR_MATRIX_H_
