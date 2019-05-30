#include <stdio.h>
extern "C"{
#include "utils.h"
}
#include "kernels.cu"

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
    int* row_ptr_array, *col_ind_array;
    double* values_array, *x_array;
    // Matrix on device
    int* row_ptr_array_d, *col_ind_array_d;
    double* values_array_d, *x_array_d;

    int num_threads         = atoi(argv[1]);
    int num_repetitions     = atoi(argv[2]);
    int flag_stdout         = atoi(argv[3]);
    char* input_filename    = argv[4];

    readMatrixFromFile(input_filename,
                        // First row of file
                       &rows, &columns, &num_of_non_zero_entries,
                        // Return variables
                       row_ptr_array, col_ind_array, values_array);
    // Init. x to 1 (in kernel)
    x_array = (double*)malloc(sizeof(double) * rows);
    memset(x_array, 1, sizeof(double) * rows);
    // cudaMemset(x_array_d, 1, sizeof(double) * rows);
    if (flag_stdout == 1)
    {
        printMatrix(rows, columns, num_of_non_zero_entries,
                row_ptr_array, col_ind_array, values_array);
        printVector(rows, x_array);
    }
    
    
    // Allocate on device
    cudaMalloc(&row_ptr_array_d, rows);
    cudaMalloc(&col_ind_array_d, columns);
    cudaMalloc(&x_array_d, rows);
    cudaMalloc(&values_array_d, rows);
    // Copy
    cudaMemcpy(row_ptr_array_d, row_ptr_array, rows, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind_array_d, col_ind_array, columns, cudaMemcpyHostToDevice);
    cudaMemcpy(values_array_d, values_array, rows, cudaMemcpyHostToDevice);
    cudaMemcpy(x_array_d, x_array, rows, cudaMemcpyHostToDevice);
    
    // Kernel invocation here
    int tmp = ceil(rows / num_threads);
    dim3 dimGrid(tmp,1);
    dim3 dimBlock(num_threads, 1);
    mmult_kernel<<<dimGrid, dimBlock>>>(rows, columns, num_of_non_zero_entries,
                                        num_repetitions,
                                        row_ptr_array_d, col_ind_array_d,
                                        values_array_d, x_array_d );
    
    // Read back from the device
//    cudaMemcpy(row_ptr_array, row_ptr_array_d, rows, cudaMemcpyDeviceToHost);
//    cudaMemcpy(col_ind_array, col_ind_array_d, columns, cudaMemcpyDeviceToHost);
//    cudaMemcpy(values_array, values_array_d, rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_array, x_array_d, rows, cudaMemcpyDeviceToHost);
    

//    cuda_hello<<<1,1>>>();
    if (flag_stdout == 1 || flag_stdout == 2)
    {
        printVector(rows, x_array);
    }
    return 0;
}
