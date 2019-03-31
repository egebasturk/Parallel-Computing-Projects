#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>

int** readDocuments(char* inputDocFilename, int dictionarySize){
#define INIT_BUFFER_DOC_NUM 100
    FILE *fptr;

    fptr = fopen(inputDocFilename, "r");
    if (fptr == NULL) {
        printf("Error reading file");
        return NULL;
    } else {
        int index = 0;
        int read_num;

        int** return_array = malloc(dictionarySize * INIT_BUFFER_DOC_NUM * sizeof(int));
        while (fscanf(fptr, "%d", &read_num) == 1)
        {
            return_array[index] = read_num;
            for (int i = 1; i < dictionarySize; ++i) {
                return_array[index + i] = ;
            }
            index++;
        }

        return return_array;
    }
}