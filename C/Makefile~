CC = clang++

CFLAGS = -g -Wall -O3 -march=native -fopenmp
LDFLAGS = -lm -lfftw3

TARGET = dustSimulation
SOURCES = $(TARGET).cpp Point.cpp Particle.cpp

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	ln $(TARGET) ../$(TARGET)

clean:
	$(RM) $(TARGET)
