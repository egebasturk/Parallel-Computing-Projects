#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
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
int main (int argc, char *argv[])
{
    MPI_Status s;
    int size, rank, i, j;

    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    if (rank == 0) // Master process
    {
        int* arr;
        int array_length = readArrayFromFile(&arr);
        printf("Array length is %d\n", array_length);

        /**
         * Send an array to the clients
         * */
        printf ("Sending data . . .\n");

        for (int l = 1; l < size; ++l)
        {
            int* arr_tmp = arr + array_length % size;
            int array_piece_length = array_length / size;
            printf ("Sending data to %d\n", l);

            MPI_Send((void *) &array_piece_length, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD); // Send length

            int offset = (array_length / size) * l * sizeof(int);

            MPI_Send((void *) arr_tmp + offset, array_length / size, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
        }
        int sum_local = 0;
        for (int m = 0; m < array_length / size + array_length % size; ++m) {
            sum_local += arr[m];
        }

        /**
         * All Reduce
         * */
        int sum_others = 0;
        printf("Process %d starting all reduce with local sum %d\n", rank, sum_local);
        MPI_Allreduce(&sum_local, &sum_others, 1,
                                       MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        printf("Process %d has total %d\n", rank, sum_others);
    }
    else
    {
        /**
         * Get info from the master
         * */
        int array_length = 0;
        MPI_Recv((void *) &array_length, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        int *arr = (int*) malloc(array_length* sizeof(int));

        MPI_Recv((void *) arr, array_length, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);


        /**
         * Make calculations and return results
         * */
        int sum_local = 0;
        for (int l = 0; l < array_length; ++l) {
            sum_local += arr[l];
        }

        //MPI_Send ((void *)&sum_local, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD);
        int sum_others = 0;
        printf("Process %d starting all reduce with local sum_local %d\n", rank, sum_local);
        MPI_Allreduce( &sum_local, &sum_others, 1,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        printf("Process %d has total %d\n", rank, sum_others);
    }
    MPI_Finalize();
    return 0;
}
