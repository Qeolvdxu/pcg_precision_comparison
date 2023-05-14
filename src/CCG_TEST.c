#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/my_crs_matrix.h"

#include "../include/CCG.h"

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

int main(int argc, char *argv[]) {
  int i, tests;

  if (argc != 3) {
    printf("ERROR: command line arguments invalid/missing\n");
    return 1;
  }
  int test_count;
  char **files = find_files("../../test_subjects/rcm", &test_count);
  printf("files found\n");
  PRECI_DT tol = (float)atof(argv[2]);
  FILE *ofile = fopen("results_CCG_TEST.csv", "w");
  int iter, maxit;

  my_crs_matrix *test;
  my_crs_matrix *precond;

  for (tests = 0; tests < test_count; tests++) {

    printf("\n%s\n", files[tests]);

    maxit = atoi(argv[1]) - 1;
    test = my_crs_read(files[tests]); //"../test_subjects/bcsstk10.mtx.crs");
    // test_RCM = rcm_reorder(test);
    precond = eye(test->n);
    PRECI_DT *b;
    PRECI_DT *x;

    my_crs_print(test);

    b = malloc(sizeof(PRECI_DT) * test->n);
    x = calloc(test->n, sizeof(PRECI_DT));

    // b vector of 1s
    for (i = 0; i < test->n; i++)
      b[i] = 1;
    // apply CG
    printf("calling cg\n");
    iter = CCG(test, precond, b, x, maxit, tol, NULL, NULL);

    fprintf(ofile, "%s,", files[tests]);
    fprintf(ofile, "%d,", iter);

    for (i = 0; i < test->n; i++)
      fprintf(ofile, "%.2e,", x[i]);
    fprintf(ofile, "\n");

    free(b);
    free(x);
  }

  free(files);
  my_crs_free(test);
  my_crs_free(precond);

  printf(" donee \n");

  return 0;
}
