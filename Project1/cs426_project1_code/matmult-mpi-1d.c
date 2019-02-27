#include <stdio.h>
#include <malloc.h>
#include <mpi.h>

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
int* convertRowWiseMatrixToColumnWise(int* arr[], int N)
{
    int* return_array = malloc(N * N * sizeof(int));
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            return_array[col] = *arr[row + N * col];
        }
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

        result_array = malloc(N * N * sizeof(int));

        printf("Array length is %d\n", arr_length1);

        printf ("Sending data . . .\n");

        for (int l = 1; l < size; ++l)
        {
            printf ("Sending data to %d\n", l);

            MPI_Send((void *) &N, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD); // Send length
            int offset = (arr_length1 / size) * l * sizeof(int);
            MPI_Send((void *) arr_length1 + N, arr_length1 / size, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
        }
        int sum_local = 0;
        for (int m = 0; m < arr_length1 / size + arr_length1 % size; ++m) {
            sum_local += array1[m];
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

    }
    MPI_Finalize();
    return 0;
}
