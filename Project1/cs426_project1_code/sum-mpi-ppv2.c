#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>
#define INPUT_ELEMENT_SIZE 2048

int readArrayFromFile(int* arr[])
{
    char* input_filename = "../test-input.txt";
    FILE *fptr;
    int* return_array = malloc(INPUT_ELEMENT_SIZE * sizeof(int));

    fptr = fopen(input_filename, "r");
    if (fptr == NULL)
    {
        printf("Error reading file");
        return 0;
    } else
    {
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
        array_length = readArrayFromFile(&arr);
        printf("Array length is %d\n", array_length);


        /**
         * Send an array to the clients
         * */
        printf("Sending data . . .\n");

        /*int array_piece_length_master = (int) ceil((double )array_length / size);
        int remainder_down_counter = (array_length % size) - 1; // -1 because master gets the first one
        int offset = array_piece_length_master * sizeof(MPI_INT);
        for (int l = 1; l < size; ++l)
        {
            int* arr_tmp = arr;

            int array_piece_length = array_length / size;
            if (remainder_down_counter > 0) {
                array_piece_length++;
                remainder_down_counter--;
            }

            printf ("Sending data to %d\n", l);
            // Send length of data
            MPI_Send((void *) &array_piece_length, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
            MPI_Send((void *) arr_tmp + offset, array_piece_length, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
            offset += array_piece_length * sizeof(MPI_INT);
        }
        int sum_local = 0;
        for (int m = 0; m < array_piece_length_master; ++m) {
            sum_local += arr[m];
        }*/
        /**
         * Calculate piece sizes and distribute remainder over them
         * */
    }
    MPI_Bcast(&array_length, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send array length for further calc
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
    MPI_Scatterv(arr, send_counts, displc, MPI_INT, recv_buffer, send_counts[0], MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int sum_local = 0;
        for (int m = 0; m < send_counts[0]; ++m) {
            sum_local += arr[m];
        }
        /*for (int k = 0; k < size; ++k) {
            printf("sc %d", send_counts[k]);
        }
        for (int k = 0; k < size; ++k) {
            printf("dis %d", displc[k]);
        }*/

        /**
         * All Reduce
         * */
        int sum_others = 0;
        printf("Process %d starting all reduce with local sum %d\n", rank, sum_local);
        MPI_Allreduce(&sum_local, &sum_others, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        printf("Process %d has total %d\n", rank, sum_others);
    }
    if (rank != 0)
    {
        /**
         * Get info from the master
         * */
        /*int array_length = 0;
        MPI_Recv((void *) &array_length, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        int *arr = (int*) malloc(array_length* sizeof(int));

        MPI_Recv((void *) arr, array_length, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);*/
        /*int* arr = (int*) malloc(size * sizeof(int));
        int recv_count;
        MPI_Scatterv(NULL, 0, NULL, MPI_INT
                ,arr, send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);*/
        /*for (int i = 0; i < send_counts[rank]; ++i) {
            printf("Process %d printed%d\n", rank, recv_buffer[i]);
        }*/

        /**
         * Make calculations and return results
         * */
        int sum_local = 0;
        for (int l = 0; l < send_counts[rank]; ++l) {
            sum_local += recv_buffer[l];
        }

        //MPI_Send ((void *)&sum_local, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD);
        int sum_others = 0;
        printf("Process %d starting all reduce with local sum_local %d\n", rank, sum_local);
        MPI_Allreduce( &sum_local, &sum_others, 1,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        printf("Process %d has total %d\n", rank, sum_others);
        free(arr);
    }
    MPI_Finalize();
    free(arr);
    free(displc);
    free(send_counts);
    free(recv_buffer);
    return 0;
}
