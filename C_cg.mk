CC = gcc

CFLAGS = -g -Wall -Wextra -pedantic -gdwarf-4
LDFLAGS = -lm

TARGET = C_cg 
SOURCES = C_cg.c my_crs_matrix.c

C_cg: C_cg.c
	$(CC) $(CFLAGS) -o C_cg $(SOURCES) $(LDFLAGS)
