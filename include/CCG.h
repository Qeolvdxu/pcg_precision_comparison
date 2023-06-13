#ifndef CCG_MATRIX_H_
#define CCG_MATRIX_H_

#include "../include/CUSTOMIZE.h"
#include "../include/my_crs_matrix.h"

// the big one
void CCG(my_crs_matrix *A, my_crs_matrix *M, C_PRECI_DT *b, double *x,
         int max_iter, C_PRECI_DT tolerance, int *iter, C_PRECI_DT *elapsed);

// cg preconditioning related functions
void ichol(my_crs_matrix *M, double *L);
void precondition(my_crs_matrix *L, double *r, double *z);

// various linear algebra functions needed for cg
double matvec_dot(my_crs_matrix *A, double *x, double *y, int n);
double dot(double *v, double *u, int n);
void matvec(my_crs_matrix *A, double *x, double *y);
double norm(int n, double *v);
#endif // CCG_MATRIX_H_
