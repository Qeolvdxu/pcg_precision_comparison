#ifndef CuCG_MATRIX_H_
#define CuCG_MATRIX_H_

// the big one
__global__ void cgkernel();

// load my_crs_matrix into a cusparse handle 
__host__ my_crs_2_cusparse(my_crs_matrix* A, cusparseHandle_t cusparseHandle);

#endif // CuCG_MATRIX_H_
