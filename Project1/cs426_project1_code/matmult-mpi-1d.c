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
    //printf("dim1:%d dim2:%d\n", dim1, dim2);
    int *result = (int*) malloc(dim1 * dim1 * sizeof(int));
    int sum_tmp = 0;
    for (int row = 0; row < dim1; ++row) {
        for (int column = 0; column < dim1; ++column) {
            for (int i = 0; i < dim2; ++i) {
                sum_tmp += mat1[row * dim2 + i] * mat2[column * dim2 + i];
            }
            //printf("col:%d/%d row:%d/%d \tsumtmp: %d\n", column, dim1, row, dim1, sum_tmp);
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
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);               // Bcast array dimension
    result_array  = malloc(N * N * sizeof(int));
    // Split based on rows. Shouldn't be a problem since expecting square matrices.
    MPI_Barrier(MPI_COMM_WORLD);
    int grid_side = N / sqrt(size); // Will be int because of project specs.
    color = rank / (N / grid_side );
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
    printf("TMP is : %d\n", tmp);
    int count_tmp[ tmp ];
    int i = 0;
    for (int j = 0; j < size; ++j) {
        if (j % (N / grid_side) == 0) {
            count_tmp[i] = j;
            //printf("%d\t", j);
            i++;
        }
    }
    //printf("%d\n", tmp);
    for (int k = 0; k < tmp; ++k) {
        printf("%d\t", count_tmp[k]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Group_incl(world_group, tmp, count_tmp, &master_world_group);
    MPI_Comm master_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, master_world_group, MASTER_TAG, &master_comm);
    if (master_comm != MPI_COMM_NULL)
        printf("Not null %d\n", rank);

    // Scatter to grid groups as rows
    // BCast the second array to group masters
    recv_buffer = (int*)malloc(N * grid_side * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);


    if (rank != 0)
        array2_colwise = malloc(N * N * sizeof(int)); // Master reads this but others don't have the buff
    if (rank % (N / grid_side) == 0)
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
        printf("Color:%d\n", color);
        printf("Scatterin %d\n", rank);
        MPI_Scatter(array1
                , N * grid_side, MPI_INT, recv_buffer
                , N * grid_side, MPI_INT, 0, master_comm);
        printf("Scatter done on %d\n", rank);
        MPI_Bcast(array2_colwise
                , N*N, MPI_INT, 0, master_comm);
        printf("Bcast to group masters done on %d\n", rank);
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
    MPI_Bcast(recv_buffer, N * grid_side, MPI_INT, 0, comm_row);
    //printf("SCATTERING TO ROWS\n");

    MPI_Scatter(array2_colwise
            , (N * grid_side), MPI_INT, recv_col_buf
            , recv_buf_size, MPI_INT, 0, comm_row);
    if (rank == 0)
    {
        for (int j = 0; j < N * grid_side; ++j) {
            printf("%d\t", recv_col_buf[j]);
        }
        printf("\n");
    }
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

    MPI_Barrier(MPI_COMM_WORLD);
    free(partial_result);
    free(recv_buffer);
    free(recv_col_buf);
    free(result_array);
    free(array2_colwise);
    if(rank == 0){
        free(array1);
        free(array2);
    }
    MPI_Finalize();
    return 0;
}
