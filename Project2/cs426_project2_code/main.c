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



// A collective comm. function
// Only master have k ids, which correspond to least k values in ascending order
// Myids: array keeps ids of documents of a process
// myvals: keep similarity val.s of the documents
// k: how many elements will be selected
// myids.size == myvals.size == k
void kreduce(int * leastk, int * myids, int * myvals, int k, int world_size, int my_rank) {

}


int main(int argc, char *argv[]) {

    int dictionarySize = 0;
    int lineCount = 0;
    char* filenameInputDoc   = argv[1];//"../documents.txt";
    char* filenameInputQuery = argv[2];// "../query.txt";

    int** documentMatrix = readDocuments( filenameInputDoc, &dictionarySize, &lineCount);
    int* queryArray      = readQuery( filenameInputQuery, dictionarySize);
    int dictionarySizeWithIDPadding = dictionarySize + 1;

    // DEBUG PRINT
    DEBUG_PRINT_INPUT
    // END DEBUG PRINT


    // Cleanup
    for (int i = 0; i < lineCount; ++i) {
        free(documentMatrix[i]);
    }
    free(documentMatrix);
    free(queryArray);

    return 0;
}