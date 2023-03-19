IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR) -g -I../src

ODIR=obj
LDIR =../lib

LIBS=-lm


DEPS =../src/FFNeuralNet.h

_OBJ = main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

unitTest: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) ../lib/floatCsvReader.o ../src/obj/FFNeuralNet.o $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o