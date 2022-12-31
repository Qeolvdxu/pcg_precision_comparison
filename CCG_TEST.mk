CC = gcc

CFLAGS = -g -Wall -Wextra -pedantic -gdwarf-4
LDFLAGS = -lm

TARGET = CCG
SOURCES = $(TARGET).c my_crs_matrix.c

all: CCG.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
