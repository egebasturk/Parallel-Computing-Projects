#include "utils.h"

int** readDocuments(char* inputDocFilename, int dictionarySize){
#define INIT_BUFFER_DOC_NUM 100
    FILE *fptr;

    fptr = fopen(inputDocFilename, "r");
    if (fptr == NULL) {
        printf("Error reading file");
        return NULL;
    } else {
        int index = 0;
        int bufferCounter = 0, bufferSize = 0;
        int readNum;

        int** return_array = malloc(dictionarySize * INIT_BUFFER_DOC_NUM * sizeof(int));
        while (fscanf(fptr, "%d:", &readNum) == 1)
        {
            if (bufferCounter * INIT_BUFFER_DOC_NUM == bufferSize){
                bufferCounter++;
                realloc(return_array, (size_t)(INIT_BUFFER_DOC_NUM * bufferCounter));
            }
            bufferSize++;
            return_array[index][0] = readNum;
            for (int i = 1; i < dictionarySize; ++i) { // Run for the part after index
                int tmp = INVALID_VALUE;
                fscanf(fptr, "%d", &tmp); // Hope input is formatted as defined in the project
                return_array[index][index + i] = tmp;
            }
            index++;
        }
        return return_array;
    }
}