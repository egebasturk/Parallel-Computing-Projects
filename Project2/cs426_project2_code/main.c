/**
 * A Parallel document search system like supervised search
 * author: Alp Ege Basturk
 *
 * each document_i is represented with a weight vector w_i
 * w_i is composed of D number of elements s.t. each of the weight values w_i,j
 * correspond to the relationship between file_i and word_j in our dictionary.
 * D is the dictionary size. D:int
 * */

#include "utils.h"

/// Subroutine to calculate similarity of single record with the given query
int calculateSimilarity(int vals[], int query[], int dictionarySize)
{
    int sim = 0;
    for (int i = 0; i < dictionarySize; ++i) {
        sim += pow(vals[i],query[i]);
    }
    return sim;
}
/// Pack related functions
/// Pack mechanism helps sorting two parallel arrays with library function
struct pack{
    int id;
    int val;
};
struct pack* packIDsAndVals(int *vals, int *ids, int size)
{
    struct pack* resultArray = malloc(sizeof(struct pack) * size);
    for (int i = 0; i < size; ++i)
    {
        resultArray[i].id = ids[i];
        resultArray[i].val = vals[i];
    }
    return resultArray;
}
struct pack* packSimilaritiesAndIds(const int *similarities, int **ids, int size
)//,struct pack** resultArray)
{
    struct pack* resultArray = (struct pack*)malloc(sizeof(struct pack) * size);
    for (int i = 0; i < size; ++i) {
        resultArray[i].id = ids[i][0];
        resultArray[i].val = similarities[i];
    }
    return resultArray;
}
void unpackArrays(struct pack* packedArray, int** idsOut, int** valsOut, int length)
{
    for (int i = 0; i < length; ++i) {
        (*idsOut)[i] = packedArray[i].id;
        (*valsOut)[i] = packedArray[i].val;
    }
}
int compareFunc(const void* a, const void* b)
{
    return ((struct pack*)a)->val > ((struct pack*)b)->val;
}
/// End of pack related functions

/// findLocalLeastk is called by each process to calculate local least k's
/// They will be merged later by k reduce
/// The ideas is: Pack -> Sort -> Truncate -> Unpack & clean excess
void findLocalLeastk(int* similarities, int** myDocumentPartMatrix,
                     int length, int k,
                     int** myIds, int** myVals)
{
    struct pack* packedArray;
    packedArray = packSimilaritiesAndIds(similarities, myDocumentPartMatrix, length);

    qsort(packedArray, length, sizeof(struct pack), compareFunc);
    unpackArrays(packedArray, myIds, myVals, length);
    /// Truncate
    int* myIdsTruncated = malloc(k * sizeof(int));
    int* myValsTruncated = malloc(k * sizeof(int));
    for (int i = 0; i < k; ++i) {
        //printf("%d\t", k);
        myIdsTruncated[i] = (*myIds)[i];
        myValsTruncated[i] = (*myVals)[i];
    }
    int* tmpPointer = *myIds;
    *myIds = myIdsTruncated;
    free(tmpPointer);
    tmpPointer = *myVals;
    *myVals = myValsTruncated;
    free(tmpPointer);
    free(packedArray);
}

/// A collective comm. function
/// Only master have k ids, which correspond to least k values in ascending order
/// Myids: array keeps ids of documents of a process
/// myvals: keep similarity val.s of the documents
/// k: how many elements will be selected
/// myids.size == myvals.size == k

/// Only the root uses least k, and initializes tmpValStorage & tmpIDStorage
/// All call MPI_Gather to gather their parts at the root
/// Root follows the same procedure as the findLocalLeastk
/// Pack -> Sort -> Truncate -> Unpack & clean excess
void kreduce(int * leastk, int * myids, int * myvals, int k, int world_size, int my_rank)
{
    // Parallel arrays
    int* tmpValStorage;
    int* tmpIDStorage;
    if(my_rank == 0)
    {
        tmpIDStorage = malloc(sizeof(int) * k * world_size);
        tmpValStorage = malloc(sizeof(int) * k * world_size);
    }
    MPI_Gather(myids, k, MPI_INT, tmpIDStorage, k, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(myvals, k, MPI_INT, tmpValStorage, k, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) // Master
    {
        struct pack* packedArray = packIDsAndVals(tmpValStorage, tmpIDStorage, k * world_size);
        qsort(packedArray, k * world_size, sizeof(struct pack), compareFunc);

        struct pack* packedArrayTruncated = malloc(sizeof(struct pack) * k);
        for (int i = 0; i < k; ++i)
        {
            packedArrayTruncated[i] = packedArray[i];
        }
        unpackArrays(packedArrayTruncated, &leastk, &myvals, k);
        free(packedArray);
        free(packedArrayTruncated);
        free(tmpValStorage);
        free(tmpIDStorage);
    }
}
int main(int argc, char *argv[])
{
    int rank, size;
    double start_seq, end_seq, start_par, end_par, start_total, end_total;

    MPI_Init(&argc, &argv);
    start_total = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    start_seq = MPI_Wtime();
    /// Master values
    int dictionarySize = atoi(argv[1]);
    int lineCount = 0;
    int k = atoi(argv[2]);
    char* filenameInputDoc;
    char* filenameInputQuery;
    int** documentMatrix;
    int* queryArray;
    int* leastk;

    if (rank == 0)
    {
        filenameInputDoc = argv[3];
        filenameInputQuery = argv[4];
        leastk = malloc(sizeof(int) * k);
        documentMatrix = readDocuments(filenameInputDoc, dictionarySize, &lineCount);
        queryArray = readQuery(filenameInputQuery, dictionarySize);
        int dictionarySizeWithIDPadding = dictionarySize + 1; // Needed for debug, don't delete

        // DEBUG PRINT
        //DEBUG_PRINT_INPUT
        // END DEBUG PRINT
        end_seq = MPI_Wtime();
        start_par = MPI_Wtime();
        /// Send dictionary size and line count to others
        for (int l = 1; l < size; ++l) {
            MPI_Send(&dictionarySize, 1, MPI_INT, l, TAG1, MPI_COMM_WORLD);
            MPI_Send(&lineCount, 1, MPI_INT, l, TAG1, MPI_COMM_WORLD);
            MPI_Send(queryArray, dictionarySize, MPI_INT, l, TAG1, MPI_COMM_WORLD);
        }
    }
    else /// Get dictionary size and line count from the root
    {
        MPI_Recv(&dictionarySize, 1, MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&lineCount, 1, MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        queryArray = malloc(dictionarySize * sizeof(int));
        MPI_Recv(queryArray, dictionarySize, MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Data portion lengths is the array stores row counts of all processes
    // Every node calculates it, redundant but simple and don't need to communicate
    int *dataPortionLengths = (int *) malloc(sizeof(int) * size);
    /// Following partitions line counts equally, then it distributes remaining in RR
    /// Data will be sent according to these values later
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
    int** myDocumentPartMatrix;
    if (rank != 0)
    {
        myDocumentPartMatrix = (int **) malloc(dataPortionLengths[rank] * sizeof(int *));
        /// Send actual data to everyone according to offsets calculated from the portions
        for (int i = 0; i < dataPortionLengths[rank]; i++)
            myDocumentPartMatrix[i] = (int *) malloc((dictionarySize + 1) * sizeof(int));
    }
    else
    {
        myDocumentPartMatrix = documentMatrix;
    }

    /// Basically Manual Broadcast
    if (rank == 0)
    {
        int** tmpRowPtr = documentMatrix + dataPortionLengths[0];
        for (int j = 1, counter = dataPortionLengths[0]; j < size; ++j) // For each other node
        {
            for (int k = 0; k < dataPortionLengths[j]; ++k)             // Send each row
            {
                MPI_Send(*tmpRowPtr, (dictionarySize + 1), MPI_INT, j, TAG1,
                         MPI_COMM_WORLD);

                tmpRowPtr++;
            }
        }
    }
    else
    {
        for (int j = 0; j < dataPortionLengths[rank]; ++j)              // Receive each row
        {
            MPI_Recv(myDocumentPartMatrix[j], (dictionarySize + 1)
                    , MPI_INT, 0, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }
    /// At this point, every node has its part
    /// Calculate similarities and find least k locally
    int* similarities = malloc(dataPortionLengths[rank] * sizeof(int));
    for (int m = 0; m < dataPortionLengths[rank]; ++m)
    {
        similarities[m] = calculateSimilarity(&myDocumentPartMatrix[m][1],
                queryArray, dictionarySize);
    }
    int* My_ids = malloc(dataPortionLengths[rank] * sizeof(int));
    int* My_vals = malloc(dataPortionLengths[rank] * sizeof(int));

    if (k > dataPortionLengths[rank])
        k = dataPortionLengths[rank];

    findLocalLeastk(similarities, myDocumentPartMatrix, dataPortionLengths[rank], k,
                    &My_ids, &My_vals);
    /// findLocalLeastk finds local least
    /// Reduce them at root and print results
    kreduce(leastk, My_ids, My_vals, k, size, rank);
    end_par = MPI_Wtime();
    end_total = MPI_Wtime();
    if (rank == 0)
    {
        printf("Sequential Part: %f ms\n", (end_seq - start_seq) * 1000);
        printf("Parallel Part: %f ms\n", (end_par - start_par) * 1000);
        printf("Total Time: %f ms\n", (end_total - start_total) * 1000);
        printf("Least k = %d ids:\n", k);
        for (int j = 0; j < k; ++j) {
            printf("%d\n", leastk[j]);
        }
    }

    /// Cleanup
    if (rank == 0)
    {
        for (int i = 0; i < lineCount; ++i) {
            free(documentMatrix[i]);
        }
        free(documentMatrix);
        free(leastk);
    } else {
        for (int i = 0; i < dataPortionLengths[rank]; ++i) {
            free(myDocumentPartMatrix[i]);
        }
        free(myDocumentPartMatrix);
    }
    free(queryArray);
    free(dataPortionLengths);
    free(similarities);
    free(My_ids);
    free(My_vals);
    MPI_Finalize();
    return 0;
}