#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "my_crs_matrix.h"

int main(int argc, char* argv[]) {
  int i, tests;
  int test_count = 18;
  my_crs_matrix *test;

  if (argc != 3)
    {
      printf("ERROR: command line arguments invalid/missing\n");
      return 1;
    }
  char *files[18] = {
    "./test_subjects/494_bus.mtx.crs",
    "./test_subjects/662_bus.mtx.crs",
    "./test_subjects/685_bus.mtx.crs",
    "./test_subjects/dwt_869.mtx.crs",
    "./test_subjects/bcsstk01.mtx.crs",
    "./test_subjects/bcsstk02.mtx.crs",
    "./test_subjects/bcsstk03.mtx.crs",
    "./test_subjects/bcsstk04.mtx.crs",
    "./test_subjects/bcsstk05.mtx.crs",
    "./test_subjects/bcsstk06.mtx.crs",
    "./test_subjects/bcsstk07.mtx.crs",
    "./test_subjects/bcsstk08.mtx.crs",
    "./test_subjects/bcsstk09.mtx.crs",
    "./test_subjects/bcsstk10.mtx.crs",
    "./test_subjects/bcsstk11.mtx.crs",
    "./test_subjects/bcsstk12.mtx.crs",
    "./test_subjects/bcsstk13.mtx.crs",
    "./test_subjects/bcsstk14.mtx.crs"
  };
  PRECI_DT tol = (float)atof(argv[2]);

  FILE *ofile = fopen("C_cg-results.csv","w");
  int iter;
  for (tests=0; tests<test_count;tests++)
    {
      printf("\n%s\n",files[tests]);

      test = my_crs_read(files[tests]);//"./test_subjects/bcsstk10.mtx.crs");
      PRECI_DT* b;
      PRECI_DT* x;

      b = malloc(sizeof(PRECI_DT) * test->n);
      x = calloc(test->n, sizeof(PRECI_DT));

      // b vector of 1s
      for (i = 0; i < test->n; i++)
	b[i] = 1;


      // apply CG

      printf("calling cg\n");
      iter = my_crs_cg(test, b, tol, atoi(argv[1])-1, x);

      free(b);
      free(x);

      fprintf(ofile,"%s,",files[tests]);
      fprintf(ofile,"%d,",iter);
      for (i = 0; i < test->n; i++)
	fprintf(ofile,"%f,",x[i]);
      fprintf(ofile,"\n");


    }

  printf(" donee \n");

  my_crs_free(test);
  return 0;
}
