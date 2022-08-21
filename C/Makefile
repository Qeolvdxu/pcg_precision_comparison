CC = clang

CFLAGS = -v -g -Wall -Wextra -pedantic -gdwarf-4
LDFLAGS = -lm

TARGET = pcg_precision_comparison
SOURCES = $(TARGET).c my_crs_matrix.c

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(MAKE) clean
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(RM) $(TARGET)
