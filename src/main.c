#define _DEFAULT_SOURCE

#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/CCG.h"
#include "../include/CuCG.h"
#include "../include/my_crs_matrix.h"
#include "../include/trisolv.h"

// call_CuCG(files[i],b,x,maxit,tol);

typedef struct {
  char precond;
  char concurrent;
  char *name;
  char *pname;
  int matrix_count;
  char **files;
  char **pfiles;
  int maxit;
  double tol;
} Data_CG;

char **find_files(const char *dir_path, int *num_files) {
  printf("DIR = %s\n", dir_path);
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



// Function to read the configuration from a config.ini file
int readConfigFile(Data_CG *data, const char *configFileName) {
  printf("test 1");
  FILE *configFile = fopen(configFileName, "r");
  if (configFile == NULL) {
    perror("Failed to open configuration file");
    return 0;
  }

  // Initialize default values
  data->precond = 'N';
  data->concurrent = 'N';
  data->name = NULL;
  data->pname = NULL;
  data->tol = 1e-7;
  data->maxit = 10000;

  char line[256];
  while (fgets(line, sizeof(line), configFile)) {
    char key[64], value[256];
    if (sscanf(line, "%63s = %255s", key, value) == 2) {
      if (strcmp(key, "precond") == 0) {
        data->precond = value[0];
      } else if (strcmp(key, "concurrent") == 0) {
        data->concurrent = value[0];
      } else if (strcmp(key, "name") == 0) {
        data->name = strdup(value);
      } else if (strcmp(key, "pname") == 0) {
        data->pname = strdup(value);
      } else if (strcmp(key, "tol") == 0) {
        data->tol = atof(value);
      } else if (strcmp(key, "maxit") == 0) {
        data->maxit = atoi(value);
      }
    }
  }

  fclose(configFile);
  return 1;
}

void *batch_CCG(void *arg) {
  Data_CG *data = (Data_CG *)arg;
  FILE *ofile = fopen("../Data/results_CCG_TEST.csv", "w");
  int k = -1; // error location
  int i, j, q;
  double *x;
  double *b;
  int iter;
  double k_twonrm = -1.0;
  double elapsed = 0.0;
  double fault_elapsed = 0.0;

  for (j = 0; j < 1; j++) {
    for (i = 0; i < data->matrix_count; i++) {
      elapsed = 0.0;
      fault_elapsed = 0.0;
      // Create Matrix struct and Precond
      my_crs_matrix *A = my_crs_read(data->files[i]);
#ifdef INJECT_ERROR
      k = rand() % A->n;
      k_twonrm = sp2nrmrow(k, A->n, A->rowptr, A->val);
#endif

      my_crs_matrix *M;
      if (data->pfiles)
        M = my_crs_read(data->pfiles[i]);

      // allocate arrays
      x = calloc(A->n, sizeof(double));
      b = malloc(sizeof(double) * A->n);
      for (q = 0; q < A->n; q++)
        b[q] = 1;

      // run cpu
      printf("CPU CG : %s", data->files[i]);
      if (data->pfiles) {
        printf("    and    %s\n", data->pfiles[i]);
        CCG(A, M, b, x, data->maxit, data->tol, &iter, &elapsed, &fault_elapsed,
            k);
      } else {
        printf("\n");
        CCG(A, NULL, b, x, data->maxit, data->tol, &iter, &elapsed,
            &fault_elapsed, k);
      }

      if (iter == 0)
        return NULL;

      elapsed -= fault_elapsed;

      if (j == 0 && i == 0)
        fprintf(ofile, "DEVICE,MATRIX,PRECISION,ITERATIONS,WALL_TIME,MEM_WALL_"
                       "TIME,FAULT_TIME,INJECT_SITE,ROW_2-NORM,"
                       "X_VECTOR\n");
      fprintf(ofile, "CPU,");
      fprintf(ofile, "%s,", data->files[i]);
      fprintf(ofile, "%s,%d,%lf,%lf,%lf,%d,%lf,", "double", iter, elapsed, 0.0,
              fault_elapsed, k, k_twonrm);
      /*printf("cpu time : %s,%d,%lf,%d,%lf \n", C_PRECI_NAME, iter, elapsed, 0,
             fault_elapsed);*/
      // printf("TOTAL C ITERATIONS: %d", iter);
      for (q = 0; q < 5; q++) {
        fprintf(ofile, "%0.10lf,", x[q]);
        // printf("%0.10lf,", x[q]);
      }
      fprintf(ofile, "\n");

      my_crs_free(A);
      if (data->pfiles)
        my_crs_free(M);
      free(b);
      free(x);

      printf(" CPU CG Test %d complete in %d iterations!\n", i, iter);
    }
    printf("\t CPU BATCH %d FINISHED!\n", j);
  }
  printf("\t\t CPU FULLY COMPLETE!\n");
  // fclose(ofile);
  return NULL;
}

void *batch_CuCG(void *arg) {
  Data_CG *data = (Data_CG *)arg;
  FILE *ofile = fopen("../Data/results_CudaCG_TEST.csv", "w");
  printf("%d matrices\n", data->matrix_count);
  int i, j, q, iter;
  int k = -1;             // error location
  double k_twonrm = -1.0; // error location
  double elapsed = 0.0;
  double mem_elapsed = 0.0;
  double fault_elapsed = 0.0;
  double *x;
  double *b;
  int n;

  for (j = 0; j < 1; j++) {
    for (i = 0; i < data->matrix_count; i++) {
      elapsed = 0.0;
      fault_elapsed = 0.0;
      mem_elapsed = 0.0;
      // get matrix size
      //  	file = fopen(data->files[i], "r");
      my_crs_matrix *A = my_crs_read(data->files[i]);
#ifdef INJECT_ERROR
      k = rand() % A->n;
      k_twonrm = sp2nrmrow(k, A->n, A->rowptr, A->val);
#endif
      n = A->n;

      // allocate arrays
      x = calloc(n, sizeof(double));
      b = malloc(sizeof(double) * n);
      for (q = 0; q < n; q++)
        b[q] = 1;

      // run gpu
      printf("GPU CG : %s", data->files[i]);
      if (data->pfiles) {
        printf("    and    %s\n", data->pfiles[i]);
        call_CuCG(data->files[i], data->pfiles[i], b, x, data->maxit,
                  (double)data->tol, &iter, &elapsed, &mem_elapsed,
                  &fault_elapsed, k);
      } else {
        printf("\n");
        call_CuCG(data->files[i], NULL, b, x, data->maxit, (double)data->tol,
                  &iter, &elapsed, &mem_elapsed, &fault_elapsed, k);
      }
      // printf("%d %lf\n", iter, elapsed);
      elapsed -= mem_elapsed + fault_elapsed;
      if (j == 0 && i == 0)
      fprintf(ofile, "GPU,");
      fprintf(ofile, "%s,", data->files[i]);
      fprintf(ofile, "%s,%d,%lf,%lf,%lf,%d,%lf,", "double", iter, elapsed,
              mem_elapsed, fault_elapsed, k, k_twonrm);
      /*printf("gpu time : %s,%d,%lf,%lf,%lf\n", "double", iter, elapsed,
             mem_elapsed, fault_elapsed);*/
      // printf("TOTAL CUDA ITERATIONS: %d", iter);
      for (q = 0; q < 5; q++) {
        fprintf(ofile, "%0.10lf,", x[q]);
        // printf("%0.10lf,", x[q]);
      }
      fprintf(ofile, "\n");

      my_crs_free(A);
      free(x);
      free(b);

      printf("  GPU CG Test %d complete in %d iterations!\n", i, iter);
    }
    printf("\t GPU BATCH %d FINISHED!\n", j);
  }
  printf("\t\t GPU FULLY COMPLETE!\n");
  fclose(ofile);
  return NULL;
}


int main(int argc, char *argv[]) {
  printf("1\n");
  srand(time(0));

  // Set initial values
  int i = 0;
  pthread_t th1;
  pthread_t th2;
  Data_CG *data;

  data = malloc(sizeof(Data_CG));

  printf("1\n");
  if (!readConfigFile(data, "../config.ini")) {
    return 1;
  }

  printf("1\n");
  data->files = find_files(data->name, &data->matrix_count);
  printf("2\n");
  int matrix_count = data->matrix_count;
  int precond_count = data->precond == 'Y' ? matrix_count : 0;

  if (matrix_count != precond_count && data->precond == 'Y') {
    printf("ERROR: number of matrices (%d) and preconditioners (%d) do not match!\n",
           matrix_count, precond_count);
    return 1;
  }

  printf("1\n");
  if (data->precond == 'Y')
    data->pfiles = find_files(data->pname, &precond_count);
  else if (data->precond == 'N')
    data->pfiles = NULL;
  else
    printf("Bad Precond Input!\n");
  printf("1\n");

  // Iteratively run conjugate gradient for each matrix
  // Runs through C implementation on host and another thread for CUDA calling

  if (data->concurrent == 'Y') {
    printf("\n\tlaunching CCG thread...");
    pthread_create(&th1, NULL, (void *(*)(void *))batch_CCG, data);
    printf("\n\tlaunching GPU CG thread...\n");
    // pthread_create(&th2, NULL, (void *(*)(void *))batch_CuCG, data);
    batch_CuCG(data);
  } else if (data->concurrent == 'N') {
    printf("\n\trunning GPU CG function...");
    batch_CuCG(data);
    printf("\n\trunning CCG function...");
    batch_CCG(data);
  } else
    printf("Bad Concurrency Input!\n");

  if (data->concurrent == 'Y') {
    pthread_join(th1, NULL);
    //  pthread_join(th2, NULL);
  }

  // Clean
  printf("cleaning memory\n");
  for (i = 0; i < matrix_count; i++) {
    free(data->files[i]);
    if (data->precond == 'Y')
      free(data->pfiles[i]);
  }
  free(data->files);
  free(data->pfiles);
  free(data);
  printf("Tests Complete!\n");

  return 0;
}
