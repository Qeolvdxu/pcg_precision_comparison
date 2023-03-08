#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "../include/my_crs_matrix.h"
#include "../include/CCG.h"
#include "../include/CuCG.h"

char **find_files(const char *dir_path, int *num_files) {
  DIR *dir = opendir(dir_path);
  struct dirent *entry;
  char **files = NULL;
  int count = 0;

  if (dir == NULL) {
    perror("opendir");
    return NULL;
  }

  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type == DT_REG) {
      files = (char **)realloc(files, sizeof(char *) * (count + 1));
      files[count] = (char *)malloc(strlen(dir_path) + strlen(entry->d_name) + 2);
      sprintf(files[count], "%s/%s", dir_path, entry->d_name);
      count++;
    }
  }

  closedir(dir);
  *num_files = count;
  return files;
}

int main(void) {

// Set inital values
  int i = 0;
  int j = 0;
  char* name;
  double tol = 0;
  int maxit = 0;
  int matrix_count = 0;
  char **files;
  my_crs_matrix *A;
  my_crs_matrix *M;
  PRECI_DT *b;
  PRECI_DT *x;
  int iter = 0;

// Collect information from user
  printf("Conjugate Gradient GPU and CPU Precision Comparison Test\n");

 //Read Directory of Matrices
   name = "../../test_subjects/norm";
  //printf("Enter the directory of matrices: ");
  //scanf("%s",name);
  files = find_files(name,&matrix_count);

 // Set answer precision tolerance 
  tol = 1e-7;
  //printf("Enter the tolerance : ");
  //scanf("%lf",&tol);

 // Stop algorithm from continuing after this many iterations
  maxit = 10000;
  //printf("Enter the maximum iterations : ");
  //scanf("%d",&maxit);

  FILE *ofile = fopen("results_CCG_TEST.csv","w");

 // Iterativly run conjugate gradient for each matrix
 // Runs through C implementation on a thread and another for CUDA calling
  for (i = 0; i < matrix_count; i++)
  {
  	printf("%s...",files[i]);
  	fflush(stdout);

  	// Create Matrix struct and Precond
  	A = my_crs_read(files[i]);
  	M = eye(A->n);

  	// Set b to 1s and x to 0s
  	x = calloc(A->n, sizeof(PRECI_DT));
  	b = malloc(sizeof(PRECI_DT)*A->n);
  	for(j=0;j<A->n;j++) b[j] = 1;
  
  	// run cpu
  	CCG(A, M, b, x, maxit, tol);
	fprintf(ofile, "CPU,");
	fprintf(ofile, "%s,",files[i]);
	for(j = 0; j < A->n; j++)
	    fprintf(ofile,"%.2e,",x[j]);
	fprintf(ofile,"\n");

	// run gpu
  	call_CuCG(files[i],b,x,maxit,tol);
	fprintf(ofile, "GPU,");
	fprintf(ofile, "%s,",files[i]);
	for(j = 0; j < A->n; j++)
	    fprintf(ofile,"%.2e,",x[j]);
	fprintf(ofile,"\n");

	//Finished
  	printf("Done! %d/%d\n",i+1,matrix_count);


  }

  // Clean
  free(x);
  free(b);
  free(files);
  my_crs_free(A);
  my_crs_free(M); 
  printf("Tests Complete!\n");

  return 0;
}
