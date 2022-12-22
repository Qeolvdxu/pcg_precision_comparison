#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#define PRECI_DT float 

// find the dot product of two vectors
__device__ static PRECI_DT dotprod(int n, PRECI_DT *v, PRECI_DT *u) {
  PRECI_DT x;
  int i;

  for (i = 0, x = 0.0; i < n; i++)
    x += v[i] * u[i];

  return x;
}

// find the norm of a vector
__device__ static PRECI_DT norm(int n, PRECI_DT *v) {
  PRECI_DT ssq, scale, absvi;
  int i;

  if (n==1) return fabs(v[0]);

  scale = 0.0;
  ssq = 1.0;

  for (i=0; i<n; i++)
    {
      if (v[i] != 0)
	{
	  absvi = fabs(v[i]);
	  if (scale < absvi)
	    {
	      ssq = 1.0 + ssq * (scale/absvi)*(scale/absvi);
	      scale = absvi;
	    }
	  else
	    ssq = ssq + (absvi/scale)*(absvi/scale);
	}
    }
  return scale * sqrt(ssq);
}

// multiply a my_crs_matrix with a vector
__device__ void csr_times_vec(int n, PRECI_DT *val, int *col, int *rowptr, PRECI_DT *v, PRECI_DT *ans, int nz) {
  int i, j;
  for (i = 0; i < n; i++)
    {
      //  printf("%d, ",i);
      ans[i] = 0;
	     for (j = rowptr[i]; j < rowptr[i+1] && j < nz; j++) {
	       /*       printf("%f : ",ans[i]);

			printf("%f :",val[j]);

	       printf("%d : ",col[j]);

	       printf("%f\n",v[col[j]]);*/



	       ans[i] += val[j] * v[i];//col[j]];
	       //printf("checkpoint");
	     }
    }

  return;
	//return 0;
}

__global__ void cgkernel(int n, int m, int nz,
			 PRECI_DT* val, int* col, int* rowptr,
			 PRECI_DT* b, PRECI_DT* x,
			 PRECI_DT* p, PRECI_DT* r, PRECI_DT* q,
			 PRECI_DT* z, PRECI_DT alpha, PRECI_DT beta)
{
  printf("%d\n",col[0]);

  int i, j, v;
  for (i=0; i < n; i++)
    {
      x[i] = 0;
      b[i] = 1;
    }

  PRECI_DT init_norm = norm(n,r);
  PRECI_DT norm_ratio = 1;

  printf("first before r = %f\n",r[0]);

  csr_times_vec(n, val,col, rowptr, x, r, nz);

  printf("first after r = %f\n",r[0]);


  for (i=0; i < n; i++)
    r[i] = b[i] - r[i];

  for (i=0; i < n; i++)
	z[i] = r[i];

  for (i=0; i < n; i++)
    p[i] = z[i];

  i=0;

  printf("is %f > %f ?\n", norm_ratio , 1e-1);

  while(i <= 2000 && norm_ratio > 1e-1)
    {
      i++;
      csr_times_vec(n, val,col, rowptr, p, q, nz);
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


      norm_ratio = norm (n,r)/init_norm;
	  
      /*printf("n=%d\tb=",n);
	for(j=0;j<n;j++)
	printf("%f\t",b[i]);*/

      printf("is (%f / %f) = %f > %f ?\n", norm(n,r), norm(n,b), norm(n,r) / norm(n,b) , 1e-6);

    }
  printf("%d iterations\n",i);

}


int main(void)
{
  int size, nz;
  int m;
  FILE *file = fopen("test_subjects/bcsstk08.mtx.crs", "r");
  int i;


  fscanf(file, "%d %d %d", &m, &size, &nz);
  PRECI_DT *h_val = (PRECI_DT*)malloc(sizeof(PRECI_DT)*nz);
  int *h_col = (int*)malloc(sizeof(int)*nz);
  int *h_rowptr = (int*)malloc(sizeof(int)*size);
  
  for (i = 0; i < size; i++)
    fscanf(file, "%d ", &h_rowptr[i]);

  for (i = 0; i < nz; i++)
    fscanf(file, "%d ", &h_col[i]);

  for (i = 0; i < nz; i++)
    fscanf(file, "%f ", &h_val[i]);


    
  //fclose(file);

  // allocate host variables for cg

  PRECI_DT *h_b = (PRECI_DT*)malloc(sizeof(PRECI_DT)*size);
  PRECI_DT *h_x = (PRECI_DT*)calloc(size, sizeof(PRECI_DT));
  PRECI_DT *h_p = (PRECI_DT*)malloc(sizeof(PRECI_DT)*size);
  PRECI_DT *h_r = (PRECI_DT*)malloc(sizeof(PRECI_DT)*size);
  PRECI_DT *h_q = (PRECI_DT*)malloc(sizeof(PRECI_DT)*size);
  PRECI_DT *h_z = (PRECI_DT*)malloc(sizeof(PRECI_DT)*size);
  PRECI_DT alpha = 0;
  PRECI_DT beta = 0;

  // set initial data values
  

  // allocate device variables for cg
  PRECI_DT *d_val;
  int *d_col;
  int *d_rowptr;

  PRECI_DT *d_b;
  PRECI_DT *d_x;
  PRECI_DT *d_p;
  PRECI_DT *d_r;
  PRECI_DT *d_q;
  PRECI_DT *d_z;
  //PRECI_DT d_alpha;
  //PRECI_DT d_beta;

  cudaMalloc((void **) &d_val, size * sizeof(PRECI_DT));
  cudaMalloc((void **) &d_col, size * sizeof(int));
cudaMalloc((void **) &d_rowptr, size * sizeof(int));

 cudaMalloc((void **) &d_b, size * sizeof(PRECI_DT));
 cudaMalloc((void **) &d_x, size * sizeof(PRECI_DT));
 cudaMalloc((void **) &d_p, size * sizeof(PRECI_DT));
 cudaMalloc((void **) &d_r, size * sizeof(PRECI_DT));
 cudaMalloc((void **) &d_q, size * sizeof(PRECI_DT));
 cudaMalloc((void **) &d_z, size * sizeof(PRECI_DT));
 //cudaMalloc((void **) &d_alpha, sizeof(PRECI_DT));
    //cudaMalloc((void **) &d_beta, sizeof(PRECI_DT));

    // copy host data to device data
 cudaMemcpy(d_val, h_val, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice); 
 cudaMemcpy(d_col, h_col, size * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_rowptr, h_rowptr, size * sizeof(int), cudaMemcpyHostToDevice); 

    cudaMemcpy(d_x, h_x, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_p, h_p, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_r, h_r, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_q, h_q, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_z, h_z, size * sizeof(PRECI_DT), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_alpha, h_alpha, sizeof(PRECI_DT), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_beta, h_beta, sizeof(PRECI_DT), cudaMemcpyHostToDevice);




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
    cudaMemcpy(h_x, d_x, sizeof(PRECI_DT)*size, cudaMemcpyDeviceToHost);

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
