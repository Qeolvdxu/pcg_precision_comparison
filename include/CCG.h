#ifndef CCG_MATRIX_H_
#define CCG_MATRIX_H_

#include "../include/CUSTOMIZE.h"
#include "../include/my_crs_matrix.h"

// the big one
void CCG(my_crs_matrix *A, my_crs_matrix *M, C_PRECI_DT *b, C_PRECI_DT *x,
         int max_iter, C_PRECI_DT tolerance, int *iter, C_PRECI_DT *elapsed);

// cg preconditioning related functions
void ichol(my_crs_matrix *M, C_PRECI_DT *L);
void precondition(my_crs_matrix *L, C_PRECI_DT *r, C_PRECI_DT *z);

// various linear algebra functions needed for cg
C_PRECI_DT matvec_dot(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y, int n);
C_PRECI_DT dot(C_PRECI_DT *v, C_PRECI_DT *u, int n);
void matvec(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y);
C_PRECI_DT norm(int n, C_PRECI_DT *v);
#endif // CCG_MATRIX_H_
