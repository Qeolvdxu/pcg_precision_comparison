#ifndef CuCG_MATRIX_H_
#define CuCG_MATRIX_H_

#include "../include/CUSTOMIZE.h"

extern void call_CuCG(char *name, char *m_name, double *h_b, double *h_x,
                      int maxit, double tol, int *iter, double *elpased,
                      double *mem_elapsed, double *fault_elapsed, int k);

#endif // CuCG_MATRIX_H_
