#define _DEFAULT_SOURCE

#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/CCG.h"
#include "../include/CUSTOMIZE.h"
#include "../include/CuCG.h"
#include "../include/my_crs_matrix.h"

// call_CuCG(files[i],b,x,maxit,tol);

typedef struct {
  int matrix_count;
  char **files;
  char **pfiles;
  int maxit;
  float tol;
} Data_CG;

char **find_files(const char *dir_path, int *num_files) {
  DIR *dir = opendir(dir_path);
  struct dirent *entry;
  char **files = NULL;
  int count = 0;
  int i, j;
  char *temp;

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

  for (i = 0; i < count - 1; i++) {
    for (j = 0; j < count - i - 1; j++) {
      if (strcmp(files[j], files[j + 1]) > 0) {
        temp = files[j];
        files[j] = files[j + 1];
        files[j + 1] = temp;
      }
    }
  }

  *num_files = count;

  return files;
}

int batch_CCG(Data_CG *data) {
  FILE *ofile = fopen("results_CCG_TEST.csv", "w");
  int i, j;
  C_PRECI_DT *x;
  C_PRECI_DT *b;
  int iter;
  double elapsed;
  printf("BATCH\n");

  for (i = 0; i < data->matrix_count; i++) {
    // Create Matrix struct and Precond
    printf("%s", data->files[i]);
    if (data->pfiles)
      printf("    and    %s\n", data->pfiles[i]);
    else
      printf("\n");
    my_crs_matrix *A = my_crs_read(data->files[i]);

    my_crs_matrix *M;
    if (data->pfiles)
      M = my_crs_read(data->pfiles[i]);
    int n = A->n;

    // allocate arrays
    x = calloc(A->n, sizeof(C_PRECI_DT));
    b = malloc(sizeof(C_PRECI_DT) * A->n);
    for (j = 0; j < A->n; j++)
      b[j] = 1;

    // run cpu
    if (data->pfiles)
      CCG(A, M, b, x, data->maxit, (C_PRECI_DT)data->tol, &iter, &elapsed);
    else
      CCG(A, NULL, b, x, data->maxit, (C_PRECI_DT)data->tol, &iter, &elapsed);

    if (i == 0)
      fprintf(ofile,
              "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,,X_VECTOR\n");
    fprintf(ofile, "CPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,,", C_PRECI_NAME, iter, elapsed);
    printf("TOTAL C ITERATIONS: %d", iter);
    for (j = 0; j < 5; j++)
      fprintf(ofile, "%0.10lf,", x[j]);
    // printf("%0.10lf,", x[j]);
    fprintf(ofile, "\n");

    my_crs_free(A);
    if (data->pfiles)
      my_crs_free(M);
    free(b);
    free(x);

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
  CUDA_PRECI_DT_HOST *x;
  CUDA_PRECI_DT_HOST *b;
  int n;

  for (i = 0; i < data->matrix_count; i++) {
    // get matrix size
    //  	file = fopen(data->files[i], "r");
    my_crs_matrix *A = my_crs_read(data->files[i]);
    printf("%s\n", data->files[i]);
    n = A->n;

    // allocate arrays
    x = calloc(n, sizeof(CUDA_PRECI_DT_HOST));
    b = malloc(sizeof(CUDA_PRECI_DT_HOST) * n);
    for (j = 0; j < n; j++)
      b[j] = 1;

    // run gpu
    call_CuCG(data->files[i], NULL, b, x, data->maxit,
              (CUDA_PRECI_DT_HOST)data->tol, &iter, &elapsed, &mem_elapsed);
    // printf("%d %lf\n", iter, elapsed);
    if (i == 0)
      fprintf(ofile,
              "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,MEM_WALL_TIME,"
              "X_VECTOR\n");
    fprintf(ofile, "GPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,%lf,", CUDA_PRECI_NAME, iter, elapsed,
            mem_elapsed);
    // printf("TOTAL CUDA ITERATIONS: %d", iter);
    for (j = 0; j < 5; j++)
      fprintf(ofile, "%0.10lf,", x[j]);
    // printf("%0.10lf,", x[j]);
    fprintf(ofile, "\n");

    my_crs_free(A);
    free(x);
    free(b);

    printf("Cuda CG Test %d complete!\n", i);
  }
  printf("\t CUDA COMPLETE!\n");
  fclose(ofile);
  return 0;
}

int main(int argc, char *argv[]) {
  // Set inital values
  int i = 0;
  char precond, concurrent;
  char *name;
  char *pname;
  double tol = 0;
  int maxit = 0;
  int matrix_count = 0;
  int precond_count = 0;
  char **files;
  char **pfiles;
  pthread_t th1, th2;
  // pthread_t th2;
  Data_CG *data;

  // Collect information from user
  printf("Conjugate Gradient GPU and CPU Precision Comparison Test\n"
         "Enter your options now, or pass them as arguments on launch\n\n");

  // Read Directory of Matrices
  name = "../../test_subjects/norm";
  pname = "../../test_subjects/precond_norm";
  // printf("Enter the directory of matrices: ");
  // scanf("%s",name);

  data = malloc(sizeof(Data_CG));

  if (argc == 1) {
    printf("Use preconditioning? (Y or N): ");
    scanf(" %c", &precond);
  } else if (argc >= 2) {
    precond = argv[1][0];
  }

  concurrent = 'Y';
  if (argc == 1) {
    printf("Run CPU and GPU concurrently? (Y or N): ");
    scanf(" %c", &concurrent);
  } else if (argc >= 3) {
    concurrent = argv[2][0];
  }

  data->files = find_files(name, &data->matrix_count);

  if (matrix_count != precond_count && precond == 'Y') {
    printf("ERROR: number of matricies (%d) and precondtioners (%d) do not "
           "match!\n",
           matrix_count, precond_count);
    return 1;
  }

  if (precond == 'Y')
    data->pfiles = find_files(pname, &precond_count);
  else if (precond == 'N')
    data->pfiles = NULL;
  else
    printf("Bad Precond Input!\n");

  // Set answer precision tolerance
  data->tol = 1e-7;
  if (argc == 1) {
    printf("Enter the tolerance : ");
    scanf(" %lf", &data->tol);
  } else if (argc >= 4) {
    data->tol = strtol(argv[3], NULL, 10);
  }

  // Stop algorithm from continuing after this many iterations
  data->maxit = 10000; // 00000;
  if (argc == 1) {
    printf("Enter the maximum iterations : ");
    scanf(" %d", &data->maxit);
  } else if (argc >= 5) {
    data->maxit = strtol(argv[4], NULL, 10);
  }

  // Iterativly run conjugate gradient for each matrix
  // Runs through C implementation on host and another thread for CUDA calling

  printf("\n\tlaunching CCG thread...");
  if (concurrent == 'Y')
    pthread_create(&th1, NULL, (void *(*)(void *))batch_CCG, data);
  else if (concurrent == 'N')
    batch_CCG(data);
  else
    printf("Bad Concurrency Input!\n");

  printf("\n\tlaunching CuCG thread...\n");
  // pthread_create(&th2, NULL, batch_CuCG, data);
  batch_CuCG(data);

  if (concurrent == 'Y')
    pthread_join(th1, NULL);
  // pthread_join(th2, NULL);

  // Clean
  printf("cleaning memory\n");
  for (i = 0; i < matrix_count; i++) {
    free(data->files[i]);
    if (precond == 'Y')
      free(data->pfiles[i]);
  }
  free(data);
  printf("Tests Complete!\n");

  return 0;
}
