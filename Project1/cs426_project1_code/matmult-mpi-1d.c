#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>

int readMatrixFromFile(int* arr[], char* input_filename, int* N)
{
    FILE *fptr;

    fptr = fopen(input_filename, "r");
    if (fptr == NULL)
    {
        printf("Error reading file");
        return 0;
    } else
    {
        int index = 0;
        int read_num;

        fscanf(fptr, "%d", N); // N: Matrix size

        int* return_array = malloc(*N * *N * sizeof(int));


        while (fscanf(fptr, "%d", &read_num) == 1)
        {
            return_array[index] = read_num;
            index++;
        }
        *arr = return_array;
        return index--;
    }
}
int* convertRowWiseMatrixToColumnWise(const int arr[], int N)
{
    int* return_array = malloc(N * N * sizeof(int));
    int index = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int tmp = arr[i + N *j];
            return_array[index] = tmp;
            index++;
        }
    }
    return return_array;
}
int main (int argc, char *argv[])
{
    MPI_Status s;
    int size, rank;

    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    if (rank == 0) // Master process
    {
        int* array1;
        int* array2;
        int* result_array;
        int N = 0;
        int arr_length1 = 0;
        int arr_length2 = 0;
        char* input_filename1 = "../test-input-matrix1.txt";
        char* input_filename2 = "../test-input-matrix2.txt";
        arr_length1 = readMatrixFromFile(&array1, input_filename1, &N);
        arr_length2 = readMatrixFromFile(&array2, input_filename2, &N);
        int* array2_colwise = convertRowWiseMatrixToColumnWise(array2, N);
        for (int i = 0; i < N * N; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", array2_colwise[i]);
        }

        result_array = malloc(N * N * sizeof(int));
        printf("Array length is %d\n", arr_length1);
        printf ("Sending data . . .\n");

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Bcast array dimension
        MPI_Bcast(array2_colwise, N*N, MPI_INT, 0, MPI_COMM_WORLD); // Bcast colwise 2nd array
        int row_count_per_process = N / size;
        //int scatter_start_offset = row_count_per_process * N * sizeof(MPI_INT);
        int* recv_buffer = (int*)malloc(N * row_count_per_process * sizeof(int));

        MPI_Scatter(array1,
            N * row_count_per_process, MPI_INT, recv_buffer,
            row_count_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
        /*for (int i = 0; i < N * row_count_per_process; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", recv_buffer[i]);
        }*/

        int* send_buffer = (int*)malloc(sizeof(int) * N * row_count_per_process);
        int tmp_sum = 0;
        int i = 0, j = 0;
        for (; j < N; ++j) {
            for (; i < row_count_per_process * N; ++i) {
                tmp_sum += recv_buffer[i] * array2_colwise[i + j*N];
            }
            send_buffer[j] = tmp_sum;
            tmp_sum = 0;
            i = 0;
        }
        MPI_Gather(send_buffer, N * row_count_per_process, MPI_INT,
                   result_array, N * row_count_per_process,
                   MPI_INT, 0, MPI_COMM_WORLD);
        printf("\nResult\n");
        for (int i = 0; i < N * N; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", result_array[i]);
        }
        /*int row = 0;
        int col = 0;
        for (int l = 1; l < size; ++l)
        {
            printf ("Sending data to %d\n", l);
            //MPI_Send((void *) &N, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD); // Send length
            //printf("Length sent\n");

            int offset = N / sqrt(size) * l * sizeof(int); // Description constraints to int
            int col = l % N;
            if (col == 0)
                row++;
            MPI_Send((void *) array1 + offset * row, N, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
            MPI_Send((void *) array2_colwise + offset * col, N, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
        }
        int sum_local = 0;
        for (int m = 0; m < N; ++m) {
            sum_local += array1[m] *array2_colwise[m];
        }

        result_array[0] = sum_local; // Save masters part
        for (int l = 1; l < size; ++l)
        {
            int tmp = 0;
            MPI_Recv((void *) &tmp, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD, &s);
        }
        printf("\nResult\n");
        for (int i = 0; i < N * N; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", result_array[i]);
        }*/
    }
    else
    {
        int N = 0;
        int* colwise_array;
        //MPI_Recv((void *) &N, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s); // Get N
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Get N, matrix dim
        colwise_array = (int*) malloc(N * N * sizeof(int));         // Create buffer
        MPI_Bcast(colwise_array, N*N, MPI_INT, 0, MPI_COMM_WORLD);  // Get the 2nd matrix

        printf("Process %d Received array with N %d\n", rank, N);

        int row_count_per_process = N / size;
        int* recv_buffer = (int*)malloc(N * row_count_per_process * sizeof(int));
        MPI_Scatter(NULL, 0, MPI_INT
                , recv_buffer, row_count_per_process * N, MPI_INT
                , 0, MPI_COMM_WORLD);
        /*for (int i = 0; i < N * row_count_per_process; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", recv_buffer[i]);
        }*/

        int* result_array = (int*)malloc(N * row_count_per_process * sizeof(int));
        int tmp_sum = 0;
        int i = 0, j = 0;
        for (; j < N; ++j) {
            for (; i < row_count_per_process * N; ++i) {
                tmp_sum += recv_buffer[i] * colwise_array[i + j*N];
            }
            result_array[j] = tmp_sum;
            tmp_sum = 0;
            i = 0;
        }

        /*for (int k = 0; k < N; ++k) {
            printf("%d ", result_array[k]);
        }*/
        MPI_Gather(result_array, row_count_per_process * N, MPI_INT
                , NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        /*for (int i = 0; i < N * row_count_per_process; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", recv_buffer[i]);
        }*/
        /*int *array1 = (int*) malloc(N* sizeof(int));
        int *array2 = (int*) malloc(N* sizeof(int));
        //printf("Process %d receiving arrays\n", rank);
        MPI_Recv((void *) array1, N, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        MPI_Recv((void *) array2, N, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        //printf("Process %d received arrays\n", rank);
*/
        /*int sum_local = 0;
        for (int l = 0; l < N; ++l) {
            sum_local += array1[l] * array2[l];
        }
        printf("Process %d sending %d to master\n", rank, sum_local);
        MPI_Send ((void *)&sum_local, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD);*/

    }
    MPI_Finalize();
    return 0;
}
