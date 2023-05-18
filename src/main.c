#define _DEFAULT_SOURCE
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/my_crs_matrix.h"

#include "../include/CCG.h"
#include "../include/CuCG.h"

// call_CuCG(files[i],b,x,maxit,tol);

typedef struct {
  int matrix_count;
  char **files;
  int maxit;
  PRECI_DT tol;
} Data_CG;

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
      files[count] =
          (char *)malloc(strlen(dir_path) + strlen(entry->d_name) + 2);
      sprintf(files[count], "%s/%s", dir_path, entry->d_name);
      count++;
    }
  }

  closedir(dir);
  *num_files = count;
  return files;
}

int batch_CCG(Data_CG *data) {
  FILE *ofile = fopen("results_CCG_TEST.csv", "w");
  int i, j;
  PRECI_DT *x;
  PRECI_DT *b;
  int m, n, z, iter;
  double elapsed;
  printf("BATCH\n");

  for (i = 0; i < data->matrix_count; i++) {
    // Create Matrix struct and Precond
    printf("%s\n", data->files[i]);
    my_crs_matrix *A = my_crs_read(data->files[i]);
    my_crs_matrix *M;
    char *m_name = NULL;
    if (m_name)
      M = my_crs_read(m_name);
    n = A->n;

    // allocate arrays
    x = calloc(A->n, sizeof(PRECI_DT));
    b = malloc(sizeof(PRECI_DT) * A->n);
    for (j = 0; j < A->n; j++)
      b[j] = 1;

    // run cpu
    CCG(A, NULL, b, x, data->maxit, data->tol, NULL, NULL, &iter, &elapsed);
    if (i == 0)
      fprintf(ofile,
              "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,,X_VECTOR\n");
    fprintf(ofile, "CPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,,", "TODO", iter, elapsed);
    for (j = 0; j < 5; j++)
      fprintf(ofile, "%0.10lf,", x[j]);
    fprintf(ofile, "\n");
    printf("C CG Test %d complete!\n", i);
  }
  printf("\t C COMPLETE!\n");
  fclose(ofile);
  return 0;
}
int batch_CuCG(Data_CG *data) {
  FILE *ofile = fopen("results_CudaCG_TEST.csv", "w");
  printf("%d matrices\n", data->matrix_count);
  int i, j, iter;
  double elapsed, mem_elapsed;
  PRECI_DT *x;
  PRECI_DT *b;
  int m, n, z;
  FILE *file;

  for (i = 0; i < data->matrix_count; i++) {
    // get matrix size
    //  	file = fopen(data->files[i], "r");
    my_crs_matrix *A = my_crs_read(data->files[i]);
    printf("%s\n", data->files[i]);
    n = A->n;

    // allocate arrays
    x = calloc(n, sizeof(PRECI_DT));
    b = malloc(sizeof(PRECI_DT) * n);
    for (j = 0; j < n; j++)
      b[j] = 1;

    // run gpu
    call_CuCG(data->files[i], NULL, b, x, data->maxit, data->tol, &iter,
              &elapsed, &mem_elapsed);
    // printf("%d %lf\n", iter, elapsed);
    if (i == 0)
      fprintf(ofile,
              "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,MEM_WALL_TIME,"
              "X_VECTOR\n");
    fprintf(ofile, "GPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,%lf,", "TODO", iter, elapsed, mem_elapsed);
    for (j = 0; j < 5; j++)
      fprintf(ofile, "%0.10lf,", x[j]);
    fprintf(ofile, "\n");
    printf("Cuda CG Test %d complete!\n", i);
  }
  printf("\t CUDA COMPLETE!\n");
  // fclose(ofile);
  return 0;
}

int main(void) {

  // Set inital values
  int i = 0;
  int j = 0;
  char *name;
  double tol = 0;
  int maxit = 0;
  int matrix_count = 0;
  char **files;
  int iter = 0;
  pthread_t th1, th2;
  Data_CG *data;

  // Collect information from user
  printf("Conjugate Gradient GPU and CPU Precision Comparison Test\n");

  // Read Directory of Matrices
  name = "../../test_subjects/norm";
  // printf("Enter the directory of matrices: ");
  // scanf("%s",name);
  files = find_files(name, &matrix_count);
  printf("%d\n", matrix_count);

  // Set answer precision tolerance
  tol = 1e-7;
  // printf("Enter the tolerance : ");
  // scanf("%lf",&tol);

  // Stop algorithm from continuing after this many iterations
  maxit = 10000; // 00000;
  // printf("Enter the maximum iterations : ");
  // scanf("%d",&maxit);

  data = malloc(sizeof(Data_CG));
  data->matrix_count = matrix_count;
  data->files = files;
  data->maxit = maxit;
  data->tol = tol;
  printf("%d\n", data->matrix_count);

  // Iterativly run conjugate gradient for each matrix
  // Runs through C implementation on a thread and another for CUDA calling
  printf("launching CCG thread...");
  pthread_create(&th1, NULL, batch_CCG, data);
  // batch_CCG(data);

  printf("launching CuCG thread...\n");
  // pthread_create(&th1, NULL, batch_CuCG, data);
  batch_CuCG(data);

  pthread_join(th1, NULL);
  // pthread_join(th2, NULL);

  // Clean
  free(files);
  printf("Tests Complete!\n");

  return 0;
}
