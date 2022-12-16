#ifndef MY_CSR_MATRIX_H_
#define MY_CSR_MATRIX_H_

#define PRECI_DT float 
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

int my_crs_times_vec(my_crs_matrix *M, PRECI_DT *v, PRECI_DT *ans);
int my_crs_cg(my_crs_matrix *M, PRECI_DT *b, PRECI_DT tol, int maxit, PRECI_DT *x);
my_crs_matrix *my_crs_read(char *name);
void my_crs_free(my_crs_matrix *M);
void my_crs_print(my_crs_matrix *M);

#endif // MY_CSR_MATRIX_H_
