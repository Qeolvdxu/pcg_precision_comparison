#ifndef CCG_MATRIX_H_
#define CCG_MATRIX_H_

#include "../include/CUSTOMIZE.h"
#include "../include/my_crs_matrix.h"

// the big one
void CCG(my_crs_matrix *A, my_crs_matrix *M, double *b, double *x, int max_iter,
         double tolerance, int *iter, double *elapsed, double *fault_elapsed);

// cg preconditioning related functions
void ichol(my_crs_matrix *M, double *L);
void forwardsub(my_crs_matrix *A, C_PRECI_DT *b, double *x);

// various linear algebra functions needed for cg
double matvec_dot(my_crs_matrix *A, C_PRECI_DT *x, C_PRECI_DT *y, int n);
double dot(double *v, double *u, int n);
void matvec(my_crs_matrix *A, double *x, double *y);
double norm(int n, double *v);
#endif // CCG_MATRIX_H_
