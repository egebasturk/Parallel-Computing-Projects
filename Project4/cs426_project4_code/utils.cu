#include "utils.cuh"
__host__
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
        
        *row_ptr_array = (int*)malloc(sizeof(int) * *num_of_non_zero_entries);
        *col_ind_array = (int*)malloc(sizeof(int) * *num_of_non_zero_entries);
        *values_array  = (double*)malloc(sizeof(double) * *num_of_non_zero_entries);
        
        
        // read lines into 3 variables line by line
        while (index < *num_of_non_zero_entries)
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
__host__
void printMatrix(int rows, int columns, int num_of_non_zero_entries,
                int* row_ptr_array, int* col_ind_array,
                double * values_array) {
    printf("%d\t%d\t%d\n", rows, columns, num_of_non_zero_entries);
    // For each row
    for (int i = 0; i < num_of_non_zero_entries; i++)
    {
        printf("%d\t%d\t%lf\n", row_ptr_array[i] + 1,
            col_ind_array[i] + 1, values_array[i]);
    }
}
__host__
void printVector(int rows, double* x_array)
{
    for (int i = 0; i < rows; i++)
    {
        printf("%lf\n", x_array[i]);
    }
    printf("\n");
}
__host__
void CUDAErrorCheck(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %d: %s.\n", msg, (int)err, cudaGetErrorName(err));
//    exit(EXIT_FAILURE);
  }
}
