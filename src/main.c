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
  double tol;
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

void *batch_CCG(void *arg) {
  Data_CG *data = (Data_CG *)arg;
  FILE *ofile = fopen("../Data/results_CCG_TEST.csv", "w");
  int i, j;
  double *x;
  double *b;
  int iter;
  double elapsed = 0.0;
  double fault_elapsed = 0.0;

  for (i = 0; i < data->matrix_count; i++) {
    elapsed = 0.0;
    fault_elapsed = 0.0;
    // Create Matrix struct and Precond
    my_crs_matrix *A = my_crs_read(data->files[i]);

    my_crs_matrix *M;
    if (data->pfiles)
      M = my_crs_read(data->pfiles[i]);

    // allocate arrays
    x = calloc(A->n, sizeof(double));
    b = malloc(sizeof(double) * A->n);
    for (j = 0; j < A->n; j++)
      b[j] = 1;

    // run cpu
    printf("CPU CG : %s", data->files[i]);
    if (data->pfiles) {
      printf("    and    %s\n", data->pfiles[i]);
      CCG(A, M, b, x, data->maxit, data->tol, &iter, &elapsed, &fault_elapsed);
    } else {
      printf("\n");
      CCG(A, NULL, b, x, data->maxit, data->tol, &iter, &elapsed,
          &fault_elapsed);
    }

    if (iter == 0)
      return NULL;

    elapsed -= fault_elapsed;

    if (i == 0)
      fprintf(ofile, "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,MEM_WALL_"
                     "TIME,FAULT_TIME,X_VECTOR\n");
    fprintf(ofile, "CPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,%d,%lf,", C_PRECI_NAME, iter, elapsed, 0,
            fault_elapsed);
    printf("cpu time : %s,%d,%lf,%d,%lf \n", C_PRECI_NAME, iter, elapsed, 0,
           fault_elapsed);
    // printf("TOTAL C ITERATIONS: %d", iter);
    for (j = 0; j < 5; j++) {
      fprintf(ofile, "%0.10lf,", x[j]);
      // printf("%0.10lf,", x[j]);
    }
    fprintf(ofile, "\n");

    my_crs_free(A);
    if (data->pfiles)
      my_crs_free(M);
    free(b);
    free(x);

    printf("CPU CG Test %d complete in %d iterations!\n", i, iter);
  }
  printf("\t CPU COMPLETE!\n");
  // fclose(ofile);
  return NULL;
}

void *batch_CuCG(void *arg) {
  Data_CG *data = (Data_CG *)arg;
  FILE *ofile = fopen("../Data/results_CudaCG_TEST.csv", "w");
  printf("%d matrices\n", data->matrix_count);
  int i, j, iter;
  double elapsed = 0.0;
  double mem_elapsed = 0.0;
  double fault_elapsed = 0.0;
  double *x;
  double *b;
  int n;

  for (i = 0; i < data->matrix_count; i++) {
    elapsed = 0.0;
    fault_elapsed = 0.0;
    mem_elapsed = 0.0;
    // get matrix size
    //  	file = fopen(data->files[i], "r");
    my_crs_matrix *A = my_crs_read(data->files[i]);
    n = A->n;

    // allocate arrays
    x = calloc(n, sizeof(double));
    b = malloc(sizeof(double) * n);
    for (j = 0; j < n; j++)
      b[j] = 1;

    // run gpu
    printf("GPU CG : %s", data->files[i]);
    if (data->pfiles) {
      printf("    and    %s\n", data->pfiles[i]);
      call_CuCG(data->files[i], data->pfiles[i], b, x, data->maxit,
                (double)data->tol, &iter, &elapsed, &mem_elapsed,
                &fault_elapsed);
    } else {
      printf("\n");
      call_CuCG(data->files[i], NULL, b, x, data->maxit, (double)data->tol,
                &iter, &elapsed, &mem_elapsed, &fault_elapsed);
    }
    // printf("%d %lf\n", iter, elapsed);
    elapsed -= mem_elapsed + fault_elapsed;
    if (i == 0)
      fprintf(ofile, "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,MEM_WALL_"
                     "TIME,FAULT_TIME,"
                     "X_VECTOR\n");
    fprintf(ofile, "GPU,");
    fprintf(ofile, "%s,", data->files[i]);
    fprintf(ofile, "%s,%d,%lf,%lf,%lf,", "double", iter, elapsed, mem_elapsed,
            fault_elapsed);
    printf("gpu time : %s,%d,%lf,%lf,%lf\n", "double", iter, elapsed,
           mem_elapsed, fault_elapsed);
    // printf("TOTAL CUDA ITERATIONS: %d", iter);
    for (j = 0; j < 5; j++) {
      fprintf(ofile, "%0.10lf,", x[j]);
      // printf("%0.10lf,", x[j]);
    }
    fprintf(ofile, "\n");

    my_crs_free(A);
    free(x);
    free(b);

    printf("GPU CG Test %d complete in %d iterations!\n", i, iter);
  }
  printf("\t GPU COMPLETE!\n");
  fclose(ofile);
  return NULL;
}

int main(int argc, char *argv[]) {
  // Set inital values
  int i = 0;
  char precond, concurrent;
  char *name;
  char *pname;
  int matrix_count = 0;
  int precond_count = 0;
  pthread_t th1;
  pthread_t th2;
  Data_CG *data;

  // Collect information from user
  printf("Conjugate Gradient GPU and CPU Precision Comparison Test\n"
         "Enter your options now, or pass them as arguments on launch\n\n");

  // Read Directory of Matrices
  name = "../test_subjects/rcm";
  pname = "../test_subjects/precond_norm";
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
  // data->matrix_count = 4;

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
    data->tol = strtof(argv[3], NULL);
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

  if (concurrent == 'Y') {
    printf("\n\tlaunching CCG thread...");
    pthread_create(&th1, NULL, (void *(*)(void *))batch_CCG, data);
    printf("\n\tlaunching GPU CG thread...\n");
    // pthread_create(&th2, NULL, (void *(*)(void *))batch_CuCG, data);
    batch_CuCG(data);
  } else if (concurrent == 'N') {
    printf("\n\trunning GPU CG function...");
    batch_CuCG(data);
    printf("\n\trunning CCG function...");
    batch_CCG(data);
  } else
    printf("Bad Concurrency Input!\n");

  if (concurrent == 'Y') {
    pthread_join(th1, NULL);
    //  pthread_join(th2, NULL);
  }

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
