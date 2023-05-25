#ifndef CCG_MATRIX_H_
#define CCG_MATRIX_H_

// the big one
void CCG(my_crs_matrix *A, my_crs_matrix *M, PRECI_DT *b, PRECI_DT *x,
         int max_iter, PRECI_DT tolerance, int *iter, double *elapsed);
// cg preconditioning related functions
void ichol(my_crs_matrix *M, double *L);
void precondition(my_crs_matrix *L, PRECI_DT *r, PRECI_DT *z);

// various linear algebra functions needed for cg
PRECI_DT matvec_dot(my_crs_matrix *A, PRECI_DT *x, PRECI_DT *y, int n);
PRECI_DT dot(PRECI_DT *v, PRECI_DT *u, int n);
void matvec(my_crs_matrix *A, PRECI_DT *x, PRECI_DT *y);
PRECI_DT norm(int n, PRECI_DT *v);
#endif // CCG_MATRIX_H_
