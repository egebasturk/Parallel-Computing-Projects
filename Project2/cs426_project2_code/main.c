/**
 * A Parallel document search system like supervised search
 * author: Alp Ege Basturk
 *
 * each document_i is represented with a weight vector w_i
 * w_i is composed of D number of elements s.t. each of the weight values w_i,j
 * correspond to the relationship between file_i and word_j in our dictionary.
 * D is the dictionary size. D:int
 * */

#include <stdio.h>
#include "utils.h"
#define TAG1 1


// A collective comm. function
// Only master have k ids, which correspond to least k values in ascending order
// Myids: array keeps ids of documents of a process
// myvals: keep similarity val.s of the documents
// k: how many elements will be selected
// myids.size == myvals.size == k
void kreduce(int * leastk, int * myids, int * myvals, int k, int world_size, int my_rank) {

}

// Subroutine to calculate similarity of single record with the query
int calculateSimilarity(int vals[], int query[], int dictionarySize)
{
    int sim = 0;
    for (int i = 0; i < dictionarySize; ++i) {
        sim += pow(vals[i],query[i]);
    }
    return sim;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /// MAster values
    int dictionarySize = 0;
    int lineCount = 0;
    char *filenameInputDoc;
    char *filenameInputQuery;
    int **documentMatrix;
    int *queryArray;

    if (rank == 0)
    {
        filenameInputDoc = argv[1];//"../documents.txt";
        filenameInputQuery = argv[2];// "../query.txt";

        documentMatrix = readDocuments(filenameInputDoc, &dictionarySize, &lineCount);
        queryArray = readQuery(filenameInputQuery, dictionarySize);
        int dictionarySizeWithIDPadding = dictionarySize + 1;

        // DEBUG PRINT
        DEBUG_PRINT_INPUT
        // END DEBUG PRINT
        /// Send dictionary size and line count to others
        for (int l = 1; l < size; ++l) {
            MPI_Send(&dictionarySize, 1, MPI_INT, l, TAG1, MPI_COMM_WORLD);
            MPI_Send(&lineCount, 1, MPI_INT, l, TAG1, MPI_COMM_WORLD);
        }
    }
    else /// Get dicsize and line count from the root
    {
        MPI_Recv(&dictionarySize, 1, MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&lineCount, 1, MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    int *dataPortionLengths = (int *) malloc(sizeof(int) * size);
    int *displc = (int *) malloc(sizeof(int) * size);
    int *recv_buffer;
    /// Following partitions line counts equally, then it distributes remaining in RR
    /// Data will be sent according to these values later
    /// Rather simple computation, everyone can compute same thing for themselves

    int remainingElementCount = lineCount;
    for (int i = 0; i < size; i++) {
        dataPortionLengths[i] = dictionarySize / size;
        remainingElementCount -= dataPortionLengths[i];
    }
    int i = 0;
    while (remainingElementCount > 0) {
        dataPortionLengths[i % size]++;
        remainingElementCount--;
        i++;
    }
    /// DEBUG
    //for (int m = 0; m < size; ++m) {
        //printf("%d\t", dataPortionLengths[m]);
    //}
    //printf("\n");
    /// END DEBUG
    displc[0] = 0; // No displacement for the initial one
    for (int j = 1; j < size; ++j) {
        displc[j] = dataPortionLengths[j - 1] + displc[j - 1];
    }
    displc[0] = 0;
    int** myDocumentPartMatrix = (int**)malloc(dataPortionLengths[rank] * sizeof(int*));

    /// Send actual data to everyone according to offsets calculated from the portions
    for (int i = 0; i < dataPortionLengths[rank]; i++)
        myDocumentPartMatrix[i] = (int*)malloc((dictionarySize + 1) * sizeof(int));


    /*if (rank == 0) {
        for (int m = 0; m < size; ++m) {
            printf("%d\t", dataPortionLengths[m]);
        }
        printf("\n");
    }*/
    if (rank == 0)
    {
        int* tmpRowPtr = documentMatrix[0];
        tmpRowPtr += dataPortionLengths[0];
        for (int j = 1; j < size; ++j)
        {
            for (int k = 0; k < dataPortionLengths[j]; ++k)
            {
                printf("%d:%d\n", j, k);
                MPI_Send(tmpRowPtr, (dictionarySize + 1), MPI_INT, j, TAG1,
                         MPI_COMM_WORLD);
                tmpRowPtr++;
            }
        }
    }
    else
    {
        for (int j = 0; j < dataPortionLengths[rank]; ++j)
        {
            MPI_Recv(myDocumentPartMatrix[j], (dictionarySize + 1)
                    , MPI_INT, 1, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }



    /// Cleanup
    if (rank == 0)
    {
        for (int i = 0; i < lineCount; ++i) {
            free(documentMatrix[i]);
        }
        free(documentMatrix);
        free(queryArray);
    }
    for (int i = 0; i < dataPortionLengths[rank]; ++i) {
        free(myDocumentPartMatrix[i]);
    }
    free(dataPortionLengths);
    free(myDocumentPartMatrix);
    free(displc);

    return 0;

}