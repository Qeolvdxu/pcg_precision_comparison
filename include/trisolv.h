#ifndef TRISOLV_H_
#define TRISOLV_H_

#include "my_crs_matrix.h"

void forwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
int s_abft_forsub(double *val, int *col, int *rowptr, int n, double *r,
                  double *y, double tol);
int s_abft_backsub(double *val, int *col, int *rowptr, int n, double *r,
                   double *y, double tol);
int s_abft_spmv(double *val, int *col, int *rowptr, int n, double *p, double *t,
                double tol);
void backwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
int isLowerTriangular(my_crs_matrix *A);
void printVector(const char *name, double *vec, int size);
void printMatrix(const char *name, my_crs_matrix *mat);
int checkSolution(my_crs_matrix *A, double *b, double *x, double tolerance);

#endif // TRISOLV_H_
