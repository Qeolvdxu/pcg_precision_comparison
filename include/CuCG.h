#ifndef CuCG_MATRIX_H_
#define CuCG_MATRIX_H_

#include "../include/CUSTOMIZE.h"

extern void call_CuCG(char *name, char *m_name, CUDA_PRECI_DT_HOST *h_b,
                      CUDA_PRECI_DT_HOST *h_x, int maxit,
                      CUDA_PRECI_DT_HOST tol, int *iter,
                      CUDA_PRECI_DT_HOST *elpased,
                      CUDA_PRECI_DT_HOST *mem_elapsed,
                      CUDA_PRECI_DT_HOST *fault_elapsed);

#endif // CuCG_MATRIX_H_
