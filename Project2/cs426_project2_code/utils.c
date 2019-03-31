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
int** readDocuments(char* inputDocFilename, int dictionarySize)
{
    FILE *fptr;
    fptr = fopen(inputDocFilename, "r");
    int lineCount = getLineCount(fptr);
    printf("Line Count: %d\n", lineCount);
    printf("D: %d\n", dictionarySize);

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