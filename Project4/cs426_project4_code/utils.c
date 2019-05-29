#include "utils.h"

void readMatrixFromFile(char* input_filename,
                        // First row of file
                        int* rows, int* columns, int* num_of_non_zero_entries,
                        // Return variables
                        int* row_ptr_array, int* col_ind_array,
                        double * values_array) {
    FILE *fptr;

    fptr = fopen(input_filename, "r");
    if (fptr == NULL) {
        printf("Error reading file");
        return;
    } else {
        int index = 0;
        // Read first row from matrix file
        fscanf(fptr, "%d %d %d", rows, columns, num_of_non_zero_entries);
        
        row_ptr_array = malloc(sizeof(int) * *rows);
        col_ind_array = malloc(sizeof(int) * *columns);
        values_array  = malloc(sizeof(double) * *num_of_non_zero_entries);
        
        
        // read lines into 3 variables line by line
        int row = 0, column = 0;
        double non_zero_val = 0.0;
        while (fscanf(fptr, "%d %d %lf", &row, &column, &non_zero_val) == 1)
        {
            row_ptr_array[index] = row;
            col_ind_array[index] = column;
            values_array[index]  = non_zero_val;
        }
    }
}
