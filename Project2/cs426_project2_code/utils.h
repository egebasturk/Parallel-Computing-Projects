#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#define INVALID_VALUE -666

int** readDocuments(char* inputDocFilename, int dictionarySize);
int* readQuery(char* inputQueryFilename, int dictionarySize);