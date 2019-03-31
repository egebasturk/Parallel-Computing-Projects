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

    int dictionarySize = 4; //TODO: change this
    int dictionarySizeWithIDPadding = dictionarySize + 1;
    int lineCount = 3;
    int** documentMatrix = readDocuments("../documents.txt", dictionarySizeWithIDPadding);
    for (int i = 0; i < lineCount; ++i) {
        for (int j = 0; j < dictionarySize; ++j) {
            printf("%d\t", documentMatrix[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < lineCount; ++i) {
        free(documentMatrix[i]);
    }
    free(documentMatrix);

    return 0;
}