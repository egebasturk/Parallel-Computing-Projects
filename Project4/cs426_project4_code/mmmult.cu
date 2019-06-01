#include <stdio.h>
#include "utils.cuh"
#include "kernels.cu"
//#define DEBUG_STOP
#define PRINT_SERIAL 1
#define PRINT_DEV_PROP 0

// Reference: Print device properties codes were taken from
// http://www.cs.fsu.edu/~xyuan/cda5125/examples/lect24/devicequery.cu

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
/*
Arguments
1. The number of threads used to compute Matrix-vector product
2. The number of repetitions and
3. An argument to print on stdout (See below).
4. Test-file name
*/
int main(int argc, char* argv[]) {
    #if PRINT_DEV_PROP
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
 
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    #ifdef DEBUG_STOP
    getchar();
    #endif
    #endif
    // Time Measurement Variables
    clock_t     timeGPUStart, timeCPUStart;
    float timeElapsedGPU = 0, timeElapsedCPU = 0,
          timeElapsedSerial = 0, timeElapsedParallel = 0,
          timeElapsedFileRead = 0;
    // Matrix meta-data
    int rows, columns, num_of_non_zero_entries;
    // Matrix
    int* row_ptr_array, *col_ind_array;
    double* values_array, *x_array, *x_array_old;
    // Matrix on device
    int* row_ptr_array_d, *col_ind_array_d;
    double* values_array_d, *x_array_d, *x_array_d_old;

    int num_threads         = atoi(argv[1]);
    int num_repetitions     = atoi(argv[2]);
    int flag_stdout         = atoi(argv[3]);
    char* input_filename    = argv[4];

    timeCPUStart = clock();
    readMatrixFromFile(input_filename,
                        // First row of file
                       &rows, &columns, &num_of_non_zero_entries,
                        // Return variables
                       &row_ptr_array, &col_ind_array, &values_array);
    // Init. x to 1 (in kernel)
    x_array = (double*)malloc(sizeof(double) * rows);
    x_array_old = (double*)malloc(sizeof(double) * rows);
    for (int i = 0; i < rows; i++)
    {
        x_array[i]     = 1.0f;
        x_array_old[i] = 1.0f;
    }
    timeElapsedCPU += clock() - timeCPUStart;
    timeElapsedSerial = timeElapsedParallel = timeElapsedCPU; // This part is common
    timeElapsedFileRead = timeElapsedSerial;
    if (flag_stdout == 1)
    {
        printf("Input Matrix:\n");
        printMatrix(rows, columns, num_of_non_zero_entries,
                row_ptr_array, col_ind_array, values_array);
        #ifdef DEBUG_STOP
        getchar();
        #endif
        printf("Initial Vector:\n");
        printVector(rows, x_array);
        #ifdef DEBUG_STOP
        getchar();
        #endif
    }
//    size_t size = num_of_non_zero_entries * sizeof(int) +
//        num_of_non_zero_entries * sizeof(int) +
//        num_of_non_zero_entries * sizeof(double) +
//        rows * sizeof(double);

    // Allocate on device
    timeGPUStart = clock();
    cudaMalloc(&row_ptr_array_d, num_of_non_zero_entries * sizeof(int));
    cudaMalloc(&col_ind_array_d, num_of_non_zero_entries * sizeof(int));
    cudaMalloc(&values_array_d, num_of_non_zero_entries * sizeof(double));
    cudaMalloc(&x_array_d, rows * sizeof(double));
    cudaMalloc(&x_array_d_old, rows * sizeof(double));
    CUDAErrorCheck("Malloc Error");
    #ifdef DEBUG_STOP
    getchar();
    #endif
    // Copy
    cudaMemcpy(row_ptr_array_d, row_ptr_array,
        num_of_non_zero_entries * sizeof(int), cudaMemcpyHostToDevice);
        
    cudaMemcpy(col_ind_array_d, col_ind_array,
        num_of_non_zero_entries * sizeof(int), cudaMemcpyHostToDevice);
        
    cudaMemcpy(values_array_d, values_array,
        num_of_non_zero_entries * sizeof(double), cudaMemcpyHostToDevice);
        
    cudaMemcpy(x_array_d_old, x_array_old,
        rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_array_d, x_array,
        rows * sizeof(double), cudaMemcpyHostToDevice);
        
    CUDAErrorCheck("Memcpy Error");
    #ifdef DEBUG_STOP
    getchar();
    #endif
        
    // Kernel invocation here
    int tmp = ceil((double)rows / num_threads);
    if (tmp == 0) tmp = 1; // 0 guard
    dim3 dimGrid(tmp,1);
    dim3 dimBlock(num_threads, 1);
    //printf("Num Threads:%d tmp:%d\n", num_threads, tmp);
    #ifdef DEBUG_STOP
    getchar();
    #endif
    for (int i = 0; i < num_repetitions; i++)
    {
        mmult_kernel<<<dimGrid, dimBlock>>>(rows, columns, num_of_non_zero_entries,
                                    num_repetitions,
                                    row_ptr_array_d, col_ind_array_d,
                                    values_array_d, x_array_d, x_array_d_old );
//        cudaMemcpy(x_array_d_old, x_array_d, rows * sizeof(double),
//                    cudaMemcpyDeviceToDevice);
        double* tmpptr = x_array_d_old;
        x_array_d_old = x_array_d;
        x_array_d = tmpptr;
        CUDAErrorCheck("Kernel Error");
    }
    x_array_d = x_array_d_old;
    #ifdef DEBUG_STOP
    getchar();
    #endif
    // Read back from the device
//    cudaMemcpy(row_ptr_array, row_ptr_array_d, rows, cudaMemcpyDeviceToHost);
//    cudaMemcpy(col_ind_array, col_ind_array_d, columns, cudaMemcpyDeviceToHost);
//    cudaMemcpy(values_array, values_array_d, rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_array, x_array_d,
        rows * sizeof(double), cudaMemcpyDeviceToHost);
    CUDAErrorCheck("Memcpy back error");
    #ifdef DEBUG_STOP
    getchar();
    #endif
    timeElapsedGPU += clock() - timeGPUStart;
    timeElapsedParallel += timeElapsedGPU;
    if (flag_stdout == 1 || flag_stdout == 2)
    {
        printf("Resulting Vector:\n");
        printVector(rows, x_array);
    }
    
    // Serial Mult.
    timeCPUStart = clock();
    for (int i = 0; i < rows; i++)
    {
        x_array[i]     = 1.0f;
        x_array_old[i] = 1.0f;
    }
    for (int i = 0; i < num_repetitions; i++)
    {
        mmult_serial(// First row of file
                           rows, columns, num_of_non_zero_entries,
                           num_repetitions,
                           row_ptr_array, col_ind_array,
                           values_array, &x_array, &x_array_old);
//        memcpy(x_array_old, x_array, rows * sizeof(double));
        double* tmpptr = x_array_old;
        x_array_old = x_array;
        x_array = tmpptr;
    }
    x_array = x_array_old;
    timeElapsedCPU += clock() - timeCPUStart;
    timeElapsedSerial += timeElapsedCPU;
    #if PRINT_SERIAL
    printf("Resulting Serial Vector:\n");
    printVector(rows, x_array);
    #endif
    
    cudaFree(x_array_d);
    cudaFree(row_ptr_array_d);
    cudaFree(col_ind_array_d);
    cudaFree(values_array_d);
    free(x_array);
    free(row_ptr_array);
    free(col_ind_array);
    free(values_array);
    printf("File Read Time: %f ms\nTotal Serial Time: %f ms\nTotal Parallel Time: %f ms\n",
            timeElapsedFileRead, timeElapsedSerial, timeElapsedParallel);
    return 0;
}
