#ifndef TRISOLV_H_
#define TRISOLV_H_

#include "my_crs_matrix.h"
void vecErrorInj_gpu(double *p, int vector_size, int k);
void vecErrorInj(double *p, int vector_size, int k);
double sp2nrmrow(int row_number, int num_rows, int *rowptr, double *val);
void forwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
int s_abft_forsub(double *val, int *col, int *rowptr, int n, double *r,
                  double *y, double tol);
int s_abft_backsub(double *val, int *col, int *rowptr, int n, double *r,
                   double *y, double tol);
void calculate_checksum(double *val, int *col, int *rowptr, int n, double *checksum);
int s_abft_spmv(double *acChecksum, int n, double *p, double *t,
                double tol);
void backwardSubstitutionCSR(my_crs_matrix *A, double *b, double *x);
int isLowerTriangular(my_crs_matrix *A);
void printVector(const char *name, double *vec, int size);
void printMatrix(const char *name, my_crs_matrix *mat);
int checkSolution(my_crs_matrix *A, double *b, double *x, double tolerance);
int abft_spmv_selective(double *val, int *col, int *rowptr, int n, double *p, double *t, double *buff, double tol, int k, int *critindex);
int compareImportance(const void *a, const void *b);
void sortByImportance(double *array1, double *array2, int length);

// Custom structure to hold values and importances together
typedef struct {
    double value;
    double importance;
} ValueWithImportance;


#endif // TRISOLV_H_
