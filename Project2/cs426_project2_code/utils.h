#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>
#define INVALID_VALUE -666
#define TAG1 1
// DEBUG
#define DEBUG_PRINT_INPUT \
    for (int i = 0; i < lineCount; ++i) { \
        for (int j = 0; j < dictionarySizeWithIDPadding; ++j) { \
            printf("%d\t", documentMatrix[i][j]); \
        } \
        printf("\n"); \
    } \
\
    for (int k = 0; k < dictionarySize; ++k) { \
        printf("%d\t", queryArray[k]); \
    } \
    printf("\n");


int** readDocuments(char* inputDocFilename, int dictionarySize, int* lineCountReturn);
int* readQuery(char* inputQueryFilename, int dictionarySize);