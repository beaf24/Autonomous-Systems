CC=gcc
CFLAGS=-Wall -O3 -lm
DEPS = operations.h
OBJ = main.o operations.o
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mapping: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean::
	rm *.o
