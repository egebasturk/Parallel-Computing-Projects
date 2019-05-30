#include <stdio.h>
#include <string.h>
#include "utils.h"

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
    int *row_ptr_array, *col_ind_array;
    double *values_array, *x_array;
    // Matrix on device
    int *row_ptr_array_d, *col_ind_array_d;
    double *values_array_d, *x_array_d;

    int num_threads = atoi(argv[1]);
    int num_repetitions = atoi(argv[2]);
    int flag_stdout = atoi(argv[3]);
    char *input_filename = argv[4];

    readMatrixFromFile(input_filename,
            // First row of file
                       &rows, &columns, &num_of_non_zero_entries,
            // Return variables
                       &row_ptr_array, &col_ind_array, &values_array);
    // Init. x to 1 (in kernel)
    x_array = (double *) malloc(sizeof(double) * rows);
    memset(x_array, 1, sizeof(double) * rows);
    // cudaMemset(x_array_d, 1, sizeof(double) * rows);
    if (flag_stdout == 1) {
        printf("Input Matrix:\n");
        printMatrix(rows, columns, num_of_non_zero_entries,
                    row_ptr_array, col_ind_array, values_array);
        printf("Initial Vector:\n");
        printVector(rows, x_array);
    }
}