#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>

__host__
void readMatrixFromFile(char* input_filename,
                        // First row of file
                        int* rows, int* columns, int* num_of_non_zero_entries,
                        // Return variables
                        int** row_ptr_array, int** col_ind_array,
                        double** values_array);
__host__
void printMatrix(int rows, int columns, int num_of_non_zero_entries,
                int* row_ptr_array, int* col_ind_array,
                double * values_array);
__host__
void printVector(int rows, double* x_array);
__host__
void CUDAErrorCheck(const char* msg);
__host__
void mmult_serial(// First row of file
                       int rows, int columns, int num_of_non_zero_entries,
                       int num_repetitions,
                        // Return variables
                       int* row_ptr_array, int* col_ind_array,
                       double* values_array, double** x_array);
