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

        int array_piece_length_master = (int) ceil((double )array_length / size);
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
        int sum = 0;
        for (int m = 0; m < array_piece_length_master; ++m) {
            sum += arr[m];
        }

        /**
         * Receive results from the clients
         * */
        printf ("Receiving data . . .\n");
        printf("Master has %d\n", sum);
        int totalsum = sum;
        for (i = 1; i < size; i++)
        {
            MPI_Recv ((void *)&j, 1, MPI_INT, i, 0xACE5, MPI_COMM_WORLD, &s);
            printf ("[%d] sent %d\n", i, j);
            totalsum += j;
            printf("Total sum is %d\n", totalsum);
        }
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
        int sum = 0;
        for (int l = 0; l < array_length; ++l) {
            sum += arr[l];
        }

        MPI_Send ((void *)&sum, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}