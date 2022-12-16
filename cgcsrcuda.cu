#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cusparse.h>

// find the dot product of two vectors
__device__ static float dotprod(int n, float *v, float *u) {
  float x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

// find the norm of a vector
__device__ static float norm(int n, float *v) {

  float norm;
  int i;

  for (i = 0, norm = 0.0; i < n; i++)
    norm += v[i] * v[i];

  norm = sqrt(norm);
  return norm;
}

// multiply a my_crs_matrix with a vector
__device__ void csr_times_vec(int n, float *val, int *col, int *rowptr, float *v, float *ans) {

  int i, j;
  for (i = 0; i < n; i++)
    ans[i] = 0;

  for (i = 0; i < n; i++) {
    for (j = rowptr[i]; j < rowptr[i]-1; j++) {
	     ans[i] = ans[i] + val[j] * v[col[j]];
	   }
	 }
  printf("%d ",ans[0]);

  return;
  //return 0;
}

__global__ void cgkernel(int n, int m, int nz,
			 float* val, int* col, int* rowptr,
			 float* b, float* x,
			 float* p, float* r, float* q,
			 float* z, float alpha, float beta)
{
  int i, j, v;
  for (i=0; i < n; i++)
      b[i] = 1;




  printf("first before r = %f\n",r[0]);
  csr_times_vec(n, val,col, rowptr, x, r);
  printf("first after r = %f\n",r[0]);



  for (i=0; i < n; i++)
	   r[i] = b[i] - r[i];


  for (i=0; i < n; i++)
    z[i] = r[i];

  for (i=0; i < n; i++)
    p[i] = z[i];

  i=0;

  printf("is (%f / %f) = %f > %f ?\n", norm(n,r), norm(n,b), norm(n,r) / norm(n,b) , 1e-6);

  while(i <= 10000 && norm(n,r) / norm(n,b) > 1e-6)

    {
      csr_times_vec(n, val,col, rowptr, p, q);
      printf("%dst after r = %f\n",i,r[0]);


      v = dotprod(n,r,z);

      alpha = v/dotprod(n,p,q);

      for (j=0; j<n; j++)
	 x[j] += alpha * p[j];
       for (j=0; j<n; j++)
	 r[j] -= alpha * q[j];
       for (j=0; j<n; j++)
	 z[j] = r[j];

      beta = dotprod(n,r,z) / v;

      for(j=0;j<n;j++)
	p[j] = z[j] + beta * p[j];
      i++;
      printf("n=%d\tb=",n);
      for(j=0;j<n;j++)
	printf("%f\t",b[i]);

      printf("is (%f / %f) = %f > %f ?\n", norm(n,r), norm(n,b), norm(n,r) / norm(n,b) , 1e-6);

    }
  printf("%d iterations\n",i);

}


int main(void)
{
  int size, nz;
  int m;
  FILE *file = fopen("bcsstk08.mtx.crs", "r");
  int i;


  fscanf(file, "%d %d %d", &m, &size, &nz);
  float *h_val = (float*)malloc(sizeof(float)*size);
  int *h_col = (int*)malloc(sizeof(int)*nz);
  int *h_rowptr = (int*)malloc(sizeof(int)*nz);
  
  for (i = 0; i < size; i++)
    fscanf(file, "%d ", &h_rowptr[i]);

  for (i = 0; i < nz; i++)
    fscanf(file, "%d ", &h_col[i]);

  for (i = 0; i < nz; i++)
    fscanf(file, "%f ", &h_val[i]);


    
  //fclose(file);

  // allocate host variables for cg

  float *h_b = (float*)malloc(sizeof(float)*size);
  float *h_x = (float*)calloc(size, sizeof(float));
  float *h_p = (float*)malloc(sizeof(float)*size);
  float *h_r = (float*)malloc(sizeof(float)*size);
  float *h_q = (float*)malloc(sizeof(float)*size);
  float *h_z = (float*)malloc(sizeof(float)*size);
  float alpha = 0;
  float beta = 0;

  // set initial data values
  

  // allocate device variables for cg
  float *d_val;
  int *d_col;
  int *d_rowptr;

  float *d_b;
  float *d_x;
  float *d_p;
  float *d_r;
  float *d_q;
  float *d_z;
  //float d_alpha;
  //float d_beta;

  cudaMalloc((void **) &d_val, size * sizeof(float));
  cudaMalloc((void **) &d_col, size * sizeof(int));
cudaMalloc((void **) &d_rowptr, size * sizeof(int));

    cudaMalloc((void **) &d_b, size * sizeof(float));
    cudaMalloc((void **) &d_x, size * sizeof(float));
    cudaMalloc((void **) &d_p, size * sizeof(float));
    cudaMalloc((void **) &d_r, size * sizeof(float));
    cudaMalloc((void **) &d_q, size * sizeof(float));
    cudaMalloc((void **) &d_z, size * sizeof(float));
    //cudaMalloc((void **) &d_alpha, sizeof(float));
    //cudaMalloc((void **) &d_beta, sizeof(float));

    // copy host data to device data
    cudaMemcpy(d_val, h_val, size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_col, h_col, size * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_rowptr, h_rowptr, size * sizeof(int), cudaMemcpyHostToDevice); 

    cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_p, h_p, size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_r, h_r, size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_q, h_q, size * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_z, h_z, size * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice);




    // Run, Time and copy data from CG
    clock_t t;
    t = clock();
    cgkernel<<<1,1>>>(size, size, nz,
		      d_val, d_col, d_rowptr,
		      d_b, d_x,
		      d_p, d_r, d_q,
		      d_z, alpha, beta);
    cudaDeviceSynchronize();
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("cg took %f seconds\n", time_taken);
    cudaMemcpy(h_x, d_x, sizeof(float)*size, cudaMemcpyDeviceToHost);

    // for (i = 0; i < size; i++)
    //printf("%f, ", h_x[i]);

    //free(h_val);
    //free(h_col);
    //free(h_rowptr);


  free(h_b);
  free(h_x);
  free(h_p);
  free(h_r);
  free(h_q);
  free(h_z);
  //free(h_alpha);
  //free(h_beta);


  cudaFree(d_val);
  cudaFree(d_col);
  cudaFree(d_rowptr);

  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_r);
  cudaFree(d_q);
  cudaFree(d_z);
  //cudaFree(d_alpha);
  //cudaFree(d_beta);
    return 0;
  }
