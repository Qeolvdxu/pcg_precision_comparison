#ifndef MY_CSR_MATRIX_H_
#define MY_CSR_MATRIX_H_

#define PRECI_DT double 
#define PRECI_S "%lf "

// compressed sparse row matrix
typedef struct {
  int n;
  int m;
  int nz;
  PRECI_DT *val;
  int *col;
  int *rowptr;
} my_crs_matrix;

// print each vector 
void my_crs_print(my_crs_matrix *M);

// create matrix
my_crs_matrix *my_crs_read(char *name);
my_crs_matrix *eye(int n);

// reorder existing matrix
my_crs_matrix *rcm_reorder(my_crs_matrix* A);
my_crs_matrix *sparse_transpose(my_crs_matrix *input);

// free the matrix from memory
void my_crs_free(my_crs_matrix *M);

#endif // MY_CSR_MATRIX_H_
