CC = clang

CFLAGS = -g -Wall
LDFLAGS = -lm

TARGET = pcg_precision_comparison
SOURCES = $(TARGET).c

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(RM) $(TARGET)
