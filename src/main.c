#include <stdio.h>
#include <stdlib.h>
#include "../include/my_crs_matrix.h"
#include "../include/CCG.h"
#include "../include/CuCG.h"

int main(void) {
  printf("Welcome to the Conjugate Gradient Precision Comparison Test!\n");

  char name[100];
  printf("Enter the .mtx file : ");
  scanf("%s",name);

  my_crs_matrix *A;
  my_crs_matrix *precond;

  call_CuCG();
  
  
  printf("Bye!\n");

  return 0;
}
