#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>

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
  PRECI_DT *b;
  PRECI_DT *x;
  int iter = 0;

 //Read Directory of Matrices
   name = "../../test_subjects/norm";
  files = find_files(name,&matrix_count);

 // Set answer precision tolerance 
  tol = 1e-7;

 // Stop algorithm from continuing after this many iterations
  maxit = 10000;

  FILE *file;
  int n;
  int m;
  int nz;

 // Iterativly run conjugate gradient for each matrix
 // Runs through C implementation on a thread and another for CUDA calling
  for (i = 0; i < matrix_count; i++)
  {
  	printf("%s...",files[i]);
  	fflush(stdout);

	
  //if ((file = fopen(files[i], "r")))
   // fscanf(file, "%d %d %d", m, n, nz);
   n = 10;

  	// Set b to 1s and x to 0s
  	x = calloc(n, sizeof(PRECI_DT));
  	b = malloc(sizeof(PRECI_DT)*n);
  	for(j=0;j<n;j++) b[j] = 1;
  
	// run gpu
  	call_CuCG(files[i],b,x,maxit,tol);
	printf("GPU,");
	printf("%s,",files[i]);
	for(j = 0; j < n; j++)
	    printf("%.2e,",x[j]);
	printf("\n");

	//Finished
  	printf("Done! %d/%d\n",i+1,matrix_count);

  }

  // Clean
  free(x);
  free(b);
  free(files);
  printf("Test Complete!\n");

  return 0;
}
