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

PRECI_DT dot(PRECI_DT *v, PRECI_DT *u, int n);
int my_crs_times_vec(my_crs_matrix *M, PRECI_DT *v, PRECI_DT *ans);
void matvec(my_crs_matrix *A, PRECI_DT* x, PRECI_DT* y);
PRECI_DT matvec_dot(my_crs_matrix *A, PRECI_DT* x, PRECI_DT* y, int n);
my_crs_matrix *my_crs_read(char *name);
void precondition(my_crs_matrix* M, PRECI_DT* x, PRECI_DT* y);
my_crs_matrix *eye(int n);

PRECI_DT norm(int n, PRECI_DT *v);

void rcm_roder(my_crs_matrix *A);
my_crs_matrix *rcm_reorder(my_crs_matrix* A);
my_crs_matrix *sparse_transpose(my_crs_matrix *input);
void my_crs_free(my_crs_matrix *M);
void my_crs_print(my_crs_matrix *M);

#endif // MY_CSR_MATRIX_H_
