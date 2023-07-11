#ifndef TRISOLV_H_
#define TRISOLV_H_

#include "my_crs_matrix.h"

void forwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
void backwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
int isLowerTriangular(my_crs_matrix *A);
void printVector(const char *name, double *vec, int size);
void printMatrix(const char *name, my_crs_matrix *mat);
int checkSolution(my_crs_matrix *A, double *b, double *x, double tolerance);

#endif // TRISOLV_H_
