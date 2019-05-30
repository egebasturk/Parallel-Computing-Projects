#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>

void readMatrixFromFile(char* input_filename,
                        // First row of file
                        int* rows, int* columns, int* num_of_non_zero_entries,
                        // Return variables
                        int* row_ptr_array, int* col_ind_array,
                        double * values_array);
void printMatrix(int rows, int columns, int num_of_non_zero_entries,
                int* row_ptr_array, int* col_ind_array,
                double * values_array);
void printVector(int rows, double* x_array);
