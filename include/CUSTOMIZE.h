#ifndef CUSTOMIZE_H_
#define CUSTOMIZE_H_

#ifdef C_DOUBLE
#define C_PRECI_DT double
#define C_PRECI_S "%lf "
#define C_PRECI_NAME "DOUBLE"
#endif
#ifdef C_SINGLE
#define C_PRECI_DT float
#define C_PRECI_S "%f "
#define C_PRECI_NAME "SINGLE"
#endif

#ifdef CUDA_DOUBLE
#define AXPY_FUN(...) cublasDaxpy(__VA_ARGS__)
#define COPY_FUN(...) cublasDcopy(__VA_ARGS__)
#define DOT_FUN(...) cublasDdot(__VA_ARGS__)
#define NORM_FUN(...) cublasDnrm2(__VA_ARGS__)
#define SCAL_FUN(...) cublasDscal(__VA_ARGS__)
#define CUDA_PRECI_DT_DEVICE CUDA_R_64F
#define CUDA_PRECI_DT_HOST double
#define CUDA_PRECI_S "%lf "
#define CUDA_PRECI_NAME "DOUBLE"
#endif

#ifdef CUDA_SINGLE
#define AXPY_FUN(...) cublasSaxpy(__VA_ARGS__)
#define COPY_FUN(...) cublasScopy(__VA_ARGS__)
#define DOT_FUN(...) cublasSdot(__VA_ARGS__)
#define NORM_FUN(...) cublasSnrm2(__VA_ARGS__)
#define SCAL_FUN(...) cublasSscal(__VA_ARGS__)
#define CUDA_PRECI_DT_DEVICE CUDA_R_32F
#define CUDA_PRECI_DT_HOST float
#define CUDA_PRECI_S "%f "
#define CUDA_PRECI_NAME "SINGLE"
#endif

#endif
