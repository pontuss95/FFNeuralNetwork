IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR) -I. -g

ODIR=obj
LDIR =../lib

LIBS=-lm

#_DEPS = floatCsvReader.h
#DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

#_OBJ = hellomake.o hellofunc.o 
_OBJ = FFNeuralNet.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/FFNeuralNet.o: FFNeuralNet.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
