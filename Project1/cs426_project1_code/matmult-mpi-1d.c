#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>
#define MASTER_TAG 666
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
int* matrixMatrixMultSerial(int *mat1, int *mat2, int dim1, int dim2) // Dim 2 is the big one. Output will be a square matrix because of project specs.
{
    printf("dim1:%d dim2:%d\n", dim1, dim2);
    int *result = (int*) malloc(dim1 * dim1 * sizeof(int));
    int sum_tmp = 0;
    for (int row = 0; row < dim1; ++row) {
        for (int column = 0; column < dim1; ++column) {
            for (int i = 0; i < dim2; ++i) {
                sum_tmp += mat1[row * dim2 + i] * mat2[column * dim2 + i];
            }
            printf("col:%d/%d row:%d/%d \tsumtmp: %d\n", column, dim1, row, dim1, sum_tmp);
            result[column + row * dim1] = sum_tmp;
            sum_tmp = 0;
        }
        //printf("OUT");
    }
    return result;
}
int main (int argc, char *argv[])
{
    MPI_Status s;
    int size, rank, N;

    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    /*
    if (rank == 0) // Master process
    {
        int* array1;
        int* array2;
        int* result_array;
        char* input_filename1 = argv[1];
        char* input_filename2 = argv[2];
        readMatrixFromFile(&array1, input_filename1, &N);
        readMatrixFromFile(&array2, input_filename2, &N);
        int* array2_colwise = convertRowWiseMatrixToColumnWise(array2, N);
        for (int i = 0; i < N * N; ++i)
        {
            if ((i % N) == 0)
                printf("\n");
            printf("%d ", array2_colwise[i]);
        }

        result_array = malloc(N * N * sizeof(int));
        printf("Array length is %d\n", N*N);
        printf ("Sending data . . .\n");

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Bcast array dimension
        MPI_Bcast(array2_colwise, N*N, MPI_INT, 0, MPI_COMM_WORLD); // Bcast colwise 2nd array
        */
        /*int current_row = 0;
        int current_col = 1;
        int current_worker_rank = 1;
        int process_count_per_node = (N * N) / size;
        for (int i = 1; i < N * N; ++i) {
            // Wrap arounds
            if ((i+1) % N == 0) {
                current_row++;
            }
            if (current_worker_rank == size) {
                current_worker_rank = 0;
            }
            int send_offset = current_worker_rank * N * process_count_per_node * sizeof(int); // Send the row
            MPI_Send(array1 + send_offset, N, MPI_INT, current_worker_rank, 666, MPI_COMM_WORLD);
        }
        */
        /*
        int row_count_per_process = N / size;
        int* recv_buffer = (int*)malloc(N * row_count_per_process * sizeof(int));

        MPI_Scatter(array1,
            N * row_count_per_process, MPI_INT, recv_buffer,
            row_count_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

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

        free(array1);
        free(array2);
        free(result_array);
        //free(send_buffer);
        //free(recv_buffer);
    }
    else
    {
        int* colwise_array;

        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Get N, matrix dim
        colwise_array = (int*) malloc(N * N * sizeof(int));         // Create buffer
        MPI_Bcast(colwise_array, N*N, MPI_INT, 0, MPI_COMM_WORLD);  // Get the 2nd matrix

        printf("Process %d Received array with N %d\n", rank, N);

        int row_count_per_process = N / size;
        int* recv_buffer = (int*)malloc(N * row_count_per_process * sizeof(int));
        MPI_Scatter(NULL, 0, MPI_INT
                , recv_buffer, row_count_per_process * N, MPI_INT
                , 0, MPI_COMM_WORLD);

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

        MPI_Gather(result_array, row_count_per_process * N, MPI_INT
                , NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

        free(colwise_array);
        free(recv_buffer);
        free(result_array);
    }
    */
    int color;
    MPI_Comm* comms;

    // Master Only
    int *array1;
    int *array2;
    int *result_array;
    int *array2_colwise;
    int* recv_buffer;
    // Master Only

    if (rank == 0) // Master process
    {
        char* input_filename1 = argv[1];
        char* input_filename2 = argv[2];
        readMatrixFromFile(&array1, input_filename1, &N);
        readMatrixFromFile(&array2, input_filename2, &N);
        array2_colwise = convertRowWiseMatrixToColumnWise(array2, N);

        printf("Array length is %d\n", N*N);
        printf ("Sending data . . .\n");
    }
    result_array  = malloc(N * N * sizeof(int));
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Bcast array dimension
    // Split based on rows. Shouldn't be a problem since expecting square matrices.
    MPI_Barrier(MPI_COMM_WORLD);
    int grid_side = N / sqrt(size); // Will be int because of project specs.
    color = rank / (grid_side * N);
    //printf("Color of %d: %d\n", rank, color);
    MPI_Comm comm_row;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_row);

    int rank_row, size_row;
    MPI_Comm_rank(comm_row, &rank_row);
    MPI_Comm_size(comm_row, &size_row);

    MPI_Group master_world_group;
    MPI_Group world_group;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int tmp = N / grid_side;
    int count_tmp[ tmp ];
    int i = 0;
    for (int j = 0; j < size; ++j) {
        if (j / (grid_side * N) == i * grid_side) {
            count_tmp[i] = j;
            //printf("%d\t", j);
            i++;
        }
    }
    /*printf("%d\n", tmp);
    for (int k = 0; k < tmp; ++k) {
        printf("%d\t", count_tmp[k]);
    }*/

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Group_incl(world_group, tmp, count_tmp, &master_world_group);
    MPI_Comm master_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, master_world_group, MASTER_TAG, &master_comm);
    //if (master_comm != MPI_COMM_NULL)
      //  printf("Not null %d\n", rank);

    // Scatter to grid groups as rows
    // BCast the second array to group masters
    recv_buffer = (int*)malloc(N * grid_side * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank_row % N == 0)
    {
        /*MPI_Comm master_world;
        int tmp = rank / (grid_side * N);
        int count_tmp[ tmp + 1];
        int i = 0;
        for (int j = 0; j < size; ++j) {
            if (j / (grid_side * N) == i) {
                count_tmp[i] = j;
                i++;
            }
        }
        */
        if (rank != 0)
            array2_colwise = malloc(N * N * sizeof(int));
        //printf("Scatterin %d\n", rank);

        MPI_Scatter(array1
                , N * grid_side, MPI_INT, recv_buffer
                , N * grid_side, MPI_INT, 0, master_comm);
        //printf("Scatter done on %d\n", rank);
        MPI_Bcast(array2_colwise
                , N*N, MPI_INT, 0, master_comm);
        //printf("Bcast to group masters done on %d\n", rank);
        MPI_Barrier(master_comm); // Delete after debug
    }
    MPI_Barrier(MPI_COMM_WORLD); // Delete after debug
    if (rank == 0)
        printf("Barrier passed\n");


    int recv_buf_size = grid_side * N;
    int* recv_col_buf = malloc( recv_buf_size * sizeof(int)); // should be same calc. here

    // Bcast the matrix part scattered to the group master to others in the row.
    // Scatter the colwise matrix which was BCasted to the group master.
    int offset = (rank_row) * N * sizeof(int);
    //printf("Comm row of %d: %d\n", rank, comm_row);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(recv_buffer + offset, N * grid_side, MPI_INT, 0, comm_row);
    //printf("SCATTERING TO ROWS\n");
    MPI_Scatter(array2_colwise
            , (N * grid_side), MPI_INT, recv_col_buf
            , recv_buf_size, MPI_INT, 0, comm_row);
    //printf("PROCESS %d SCATTERED TO ROWS\n", rank);
    int* partial_result;
    //printf("Grid size is: %d\n", grid_side);
    partial_result = matrixMatrixMultSerial(recv_buffer, recv_col_buf, grid_side, N);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(partial_result, grid_side * grid_side, MPI_INT
            , result_array, grid_side * grid_side, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        char* output_filename = argv[3];
        FILE *fptr;
        fptr = fopen(output_filename, "w");
        fprintf(fptr, "%d\n", N);
        for (int i = 0; i < N * N; ++i)
        {
            if ((i % N) == 0 && i != 0)
                fprintf(fptr, "\n");
            fprintf(fptr, "%d ", result_array[i]);
        }
    }



    free(partial_result);
    free(recv_buffer);
    free(recv_col_buf);
    free(result_array);
    if(rank == 0){
        free(array1);
        free(array2);
        free(array2_colwise);
    }
    MPI_Finalize();

    return 0;
}
