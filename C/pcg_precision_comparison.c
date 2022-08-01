#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct my_csr_matrix {
  int n;
  int m;
  int nz;
  double *val;
  int* col;
  int* rowptr;
} my_csr_matrix;

/* static double norm(int n, double *v) {
   double norm;
  int i;

  for (i=0, norm=0.0; i<n; i++) norm += v[i]*v[i];

  norm = sqrt(norm);
  return norm;
  }*/

my_csr_matrix *csrread(char *name) {
  struct my_csr_matrix *M = malloc(sizeof(struct my_csr_matrix));
  FILE *file = fopen(name, "r");
  int i;

  fscanf(file, "%d %d %d", &M->m, &M->n, &M->nz);
  M->val = malloc(sizeof(double) * M->nz);
  M->col = malloc(sizeof(int) * M->nz);
  M->rowptr = malloc(sizeof(int) * M->n);

  for (i = 0; i < M->n; i++)
    fscanf(file, "%d ", &M->rowptr[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, "%d ", &M->col[i]);
  for (i = 0; i < M->nz; i++)
    fscanf(file, "%lf ", &M->val[i]);

  fclose(file);
  return M;
}

  void free_matrix(my_csr_matrix * M) {
    free(M->val);
    free(M->col);
    free(M->rowptr);
    free(M);

    return;
  }

  void print_csr(my_csr_matrix *M) {
    int i = 0;

    printf("rowptr,");
    for (i = 0; i < M->n; i++)
      printf("%d, ", M->rowptr[i]);
    printf("\n\n");

    printf("index,");
    for (i = 0; i < M->nz; i++)
      printf("%d, ", M->col[i]);
    printf("\n\n");

    printf("values,");
    for (i = 0; i < M->nz; i++)
      printf("%lf, ", M->val[i]);
    printf("\n\n");


  }
  int main(void) {

    struct my_csr_matrix *test = csrread("./test_subjects/bcsstk04.mtx.crs");
    print_csr(test);
    free_matrix(test);

    return 0;
  }
