#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>
#define INPUT_ELEMENT_SIZE 2048

int readArrayFromFile(int* arr[], char* input_filename)
{
    //char* input_filename = "../input";
    FILE *fptr;
    int* return_array;// = malloc(INPUT_ELEMENT_SIZE * sizeof(int));
    int line_count = 0;

    fptr = fopen(input_filename, "r");
    if (fptr == NULL)
    {
        printf("Error reading file");
        return 0;
    } else {
        int index = 0;
        int read_num;
        while (fscanf(fptr, "%d", &read_num) == 1) {
            line_count++;
        }
    }
    fptr = fopen(input_filename, "r");
    return_array = malloc(line_count * sizeof(int));
    int index = 0;
    int read_num;
    while (fscanf(fptr, "%d", &read_num) == 1)
    {
        return_array[index] = read_num;
        index++;
    }
    *arr = return_array;
    return index--;
}
int main (int argc, char *argv[]) {
    MPI_Status s;
    int size, rank;
    int array_length;
    int *arr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) // Master process
    {
        array_length = readArrayFromFile(&arr, argv[1]);
        //printf("Array length is %d\n", array_length);
    }

    MPI_Bcast(&array_length, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send array length for further calc
    /**
     * Following part handles if there will be idle processes
     * when data is not divisible that much. I.e. removes processes
     * which will get no data
     * */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm new_world;
    if (array_length < size) {
        MPI_Group mpi_world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

        MPI_Group new_group;
        int ranges[3] = {array_length, size - 1, 1}; // first rank, last rank, stride
        MPI_Group_range_excl(mpi_world_group, 1, ranges, &new_group);


        MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_world);


        if (new_world == MPI_COMM_NULL) {
            free(arr);
            MPI_Finalize();
            return 0;
        }
    } else
        new_world = MPI_COMM_WORLD;

    int remainder = array_length % size;
    int *send_counts = (int *) malloc(sizeof(int) * size);
    int *displc = (int *) malloc(sizeof(int) * size);
    int *recv_buffer;

    int remaining_element_count = array_length;
    for (int i = 0; i < size; i++) {
        send_counts[i] = array_length / size;
        remaining_element_count -= send_counts[i];
    }
    int i = 0;
    while (remaining_element_count > 0) {
        send_counts[i % size]++;
        remaining_element_count--;
        i++;
    }
    displc[0] = 0; // No displacement for the initial one
    for (int j = 1; j < size; ++j) {
        displc[j] = send_counts[j - 1] + displc[j - 1];
    }
    displc[0] = 0;

    recv_buffer = (int *) malloc(sizeof(int) * send_counts[0]);
    MPI_Scatterv(arr, send_counts, displc, MPI_INT, recv_buffer, send_counts[0], MPI_INT, 0, new_world);
    if (rank == 0) {
        int sum_local = 0;
        for (int m = 0; m < send_counts[0]; ++m) {
            sum_local += arr[m];
        }

        /**
         * All Reduce
         * */
        int sum_others = 0;
        //printf("Process %d starting all reduce with local sum %d\n", rank, sum_local);
        MPI_Allreduce(&sum_local, &sum_others, 1,
                      MPI_INT, MPI_SUM, new_world);
        //printf("Process %d has total %d\n", rank, sum_others);
        printf("%d\n", sum_others);
    }
    if (rank != 0)
    {
        /**
         * Make calculations and return results
         * */
        int sum_local = 0;
        for (int l = 0; l < send_counts[rank]; ++l) {
            sum_local += recv_buffer[l];
        }

        int sum_others = 0;
        //printf("Process %d starting all reduce with local sum_local %d\n", rank, sum_local);
        MPI_Allreduce( &sum_local, &sum_others, 1,
                MPI_INT, MPI_SUM, new_world);
        //printf("Process %d has total %d\n", rank, sum_others);
    }
    MPI_Finalize();
    free(arr);
    free(displc);
    free(send_counts);
    free(recv_buffer);
    return 0;
}
