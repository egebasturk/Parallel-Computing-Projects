#ifndef UTILS
#define UTILS
#include "utils.cuh"
#endif


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

// One thread per matrix row
__global__ void mmult_kernel(// First row of file
                       int rows, int columns, int num_of_non_zero_entries,
                       int num_repetitions,
                        // Return variables
                       int* row_ptr_array_d, int* col_ind_array_d,
                       double* values_array_d, double* x_array_d) {
    printf("Thread on GPU: BlockDim.x:%d blockIdx.x:%d threadIdx.x:%d\n"
                                      , blockDim.x, blockIdx.x, threadIdx.x);
    for (int i = 0; i < num_repetitions; i++)
    {
        // Iteration code
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row < rows)
        {
            float tmp_product;
            int row_start = row_ptr_array_d[row];
            int row_end   = row_ptr_array_d[row + 1];
            
            // Iterate over the sparse row
            for (int j = row_start; j < row_end; j++)
                tmp_product += values_array_d[j] * x_array_d[col_ind_array_d[j]];
            x_array_d[row] += tmp_product;
        }
    }
}
