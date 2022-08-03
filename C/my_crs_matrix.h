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

double *my_crs_times_vec(my_crs_matrix *M, double *v);
double *my_crs_cg(my_crs_matrix *M, PRECI_DT *b, double tol, int maxit);
my_crs_matrix *my_crs_read(char *name);
void my_crs_free(my_crs_matrix *M);
void my_crs_print(my_crs_matrix *M);

#endif // MY_CSR_MATRIX_H_
