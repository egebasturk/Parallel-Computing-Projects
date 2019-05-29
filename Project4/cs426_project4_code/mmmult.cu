#include <stdio.h>
extern "C"{
#include "utils.h"
}
/*__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}*/

/*
Arguments
1. The number of threads used to compute Matrix-vector product
2. The number of repetitions and
3. An argument to print on stdout (See below).
4. Test-file name
*/
int main(int argc, char* argv[]) {
    // Matrix meta-data
    int rows, columns, num_of_non_zero_entries;
    // Matrix
    int* row_ptr_array, *col_ind_array, *x_array;
    double* values_array;

    int num_threads         = atoi(argv[1]);
    int num_repetitions     = atoi(argv[2]);
    int flag_stdout         = atoi(argv[3]);
    char* input_filename    = argv[4];

    readMatrixFromFile(input_filename,
            // First row of file
                       &rows, &columns, &num_of_non_zero_entries,
            // Return variables
                       row_ptr_array, col_ind_array, values_array);
    x_array = (int*)malloc(sizeof(int) * rows);

//    cuda_hello<<<1,1>>>();
    return 0;
}
