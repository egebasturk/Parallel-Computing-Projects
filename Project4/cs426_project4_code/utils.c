#include "utils.h"

void readMatrixFromFile(char* input_filename,
                        // First row of file
                        int* rows, int* columns, int* num_of_non_zero_entries,
                        // Return variables
                        int** row_ptr_array, int** col_ind_array,
                        double** values_array) {
    FILE *fptr;

    fptr = fopen(input_filename, "r");
    if (fptr == NULL) {
        printf("Error reading file");
        return;
    } else {
        int index = 0;
        int row = 0, column = 0, tmp = 0;
        double non_zero_val = 0.0;
        // Read first row from matrix file
        fscanf(fptr, "%d %d %d", &row, &column, &tmp);
        *rows = row;
        *columns = column;
        *num_of_non_zero_entries = tmp;

        *row_ptr_array = (int*)malloc(sizeof(int) * *rows);
        *col_ind_array = (int*)malloc(sizeof(int) * *columns);
        *values_array  = (double*)malloc(sizeof(double) * *num_of_non_zero_entries);

        
        // read lines into 3 variables line by line
        while (index < *rows)
        {
            fscanf(fptr, "%d", &row);
            fscanf(fptr, "%d", &column);
            fscanf(fptr, "%lf", &non_zero_val);
        
            // -1 to make indices start from 0
            (*row_ptr_array)[index] = row - 1;
            (*col_ind_array)[index] = column - 1;
            (*values_array)[index]  = non_zero_val;
            index++;
        }
    }
}
void printMatrix(int rows, int columns, int num_of_non_zero_entries,
                int* row_ptr_array, int* col_ind_array,
                double * values_array) {
    printf("%d\t%d\t%d\n", rows, columns, num_of_non_zero_entries);
    // For each row
    for (int i = 0; i < rows; i++)
    {
        int row_start = row_ptr_array[i];
        int row_end   = row_ptr_array[i + 1];
        for (int j = row_start; j < row_end; j++)
        {
            printf("I:%d\tJ:%d\n", i, j);
            printf("%d\t%d\t%lf\n",
                row_ptr_array[i], col_ind_array[j], values_array[j]);
        }
    }
}
void printVector(int rows, double* x_array)
{
    for (int i = 0; i < rows; i++)
    {
        printf("%lf\t", x_array[i]);
    }
    printf("\n");
}
