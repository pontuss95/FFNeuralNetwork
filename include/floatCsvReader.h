#ifndef FLOAT_CSV_READER

#define FLOAT_CSV_READER

#include <stdio.h>
typedef struct Array
{
    //**Array is pointers to arrays of every row

    double **Array;
    unsigned int numCols;
    unsigned int numRows;
} Array;

Array *readCsv(FILE *);

#endif