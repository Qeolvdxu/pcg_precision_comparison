#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "my_crs_matrix.h"

int main(void) {
  int i;
  my_crs_matrix *test = my_crs_read("./test_subjects/test.mtx.crs");
  double *b = malloc(sizeof(double) * test->n);
  double *ans;
  my_crs_print(test);

  // b vector of 1s
  for (i = 0; i < test->n; i++)
    b[i] = 1;

  // apply CG
  ans = my_crs_cg(test, b, 1e-6, 8000);
  printf("\nans = ");
  for (i = 0; i < test->n; i++)
    printf("%lf, ", ans[i]);
  printf("\n");

  
  // free
  my_crs_free(test);
  free(b);
  free(ans);
  return 0;
}
