#ifndef CuCG_MATRIX_H_
#define CuCG_MATRIX_H_

#define PRECI_DT double

extern void call_CuCG(char* name, PRECI_DT* h_b, PRECI_DT* h_x, int maxit, PRECI_DT tol);

#endif // CuCG_MATRIX_H_
