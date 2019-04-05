#include <memory.h>
#include "utils.h"


int getLineCount(FILE* fptr)
{
    int readNum;
    int count = 0;
    for (char c = getc(fptr); c != EOF; c = getc(fptr))
        if (c == '\n')
            count++;
    rewind(fptr); // Don't effect the location
    if ( count != 0 )
        count++;
    return count;
}
int** readDocuments(char* inputDocFilename, int* dictionarySizeReturn, int* lineCountReturn)
{
    FILE *fptr;
    fptr = fopen(inputDocFilename, "r");
    int lineCount = getLineCount(fptr);
    int dictionarySize = 0;
    *lineCountReturn = lineCount;

    /// Read first line to determine directory size
    char* readBuffer = malloc(1024 * sizeof(char));
    fgets(readBuffer, 1024, fptr);
    //printf("The line is: %s\n", readBuffer);
    fseek(fptr, 0L, SEEK_SET); // Reset file pointer to start of the file

    int tmp;
    for (int j = 0; j < 1024; ++j) {
        if (readBuffer[j] == ' ')
            dictionarySize++;
    }
    free(readBuffer);
    *dictionarySizeReturn = dictionarySize;

    //printf("Line Count: %d\n", lineCount);
    //printf("D: %d\n", dictionarySize);
    /// END Read first line to determine directory size

    if (fptr == NULL)
    {
        printf("Error reading Documents file");
        return NULL;
    }
    else
    {
        int readNum;

        int** returnArray = (int**)malloc(lineCount * sizeof(int*));
        for (int i = 0; i < lineCount; i++)
            returnArray[i] = (int*)malloc((dictionarySize + 1) * sizeof(int));

        for (int index = 0; index < lineCount; ++index)
        {
            fscanf(fptr, "%d:", &readNum); // Hope input is formatted as defined in the project
            returnArray[index][0] = readNum;

            for (int i = 1; i <= dictionarySize; i++) // Run for the part after index
            {
                int tmp = INVALID_VALUE;
                fscanf(fptr, "%d", &tmp);
                returnArray[index][i] = tmp;
            }
        }
        return returnArray;
    }
}
int* readQuery(char* inputQueryFilename, int dictionarySize)
{
    FILE *fptr;
    fptr = fopen(inputQueryFilename, "r");
    int lineCount = getLineCount(fptr);
    int *returnArray;

    if (fptr == NULL)
    {
        printf("Error reading Query file");
        return NULL;
    }
    else
    {
        returnArray = (int *) malloc(dictionarySize * sizeof(int));
        for (int i = 0; i < dictionarySize; i++)
        {
            int tmp = INVALID_VALUE;
            fscanf(fptr, "%d", &tmp);
            returnArray[i] = tmp;
        }
    }
    return returnArray;
}