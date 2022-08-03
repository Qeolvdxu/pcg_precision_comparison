CC = clang

CFLAGS = -g -Wall -Wextra -pedantic
LDFLAGS = -lm

TARGET = pcg_precision_comparison
SOURCES = $(TARGET).c my_crs_matrix.c

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(MAKE) clean
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(RM) $(TARGET)
