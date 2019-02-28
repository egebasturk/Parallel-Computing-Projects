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
        int* array2_colwise = convertRowWiseMatrixToColumnWise(array2, N);

        result_array = malloc(N * N * sizeof(int));
        printf("Array length is %d\n", arr_length1);
        printf ("Sending data . . .\n");

        for (int l = 1; l < size; ++l)
        {
            printf ("Sending data to %d\n", l);

            MPI_Send((void *) &N, 1, MPI_INT, l, 0xACE5, MPI_COMM_WORLD); // Send length

            int offset = N * l * sizeof(int);
            MPI_Send((void *) array1 + offset, N, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
            MPI_Send((void *) array2_colwise + offset, N, MPI_INT, l, 0xACE5, MPI_COMM_WORLD);
        }
        int sum_local = 0;
        for (int m = 0; m < N; ++m) {
            sum_local += array1[m] *array2_colwise[m];
        }
        /**
         * Receive data
         * */
        result_array[0] = sum_local; // Save masters part
        for (int l = 1; l < size; ++l)
        {
            MPI_Recv((void *) &N, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        }
    }
    else
    {
        int N = 0;
        MPI_Recv((void *) &N, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        int *array1 = (int*) malloc(N* sizeof(int));
        int *array2 = (int*) malloc(N* sizeof(int));
        MPI_Recv((void *) array1, N, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);
        MPI_Recv((void *) array2, N, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD, &s);

        int sum_local = 0;
        for (int l = 0; l < N; ++l) {
            sum_local += array1[l] * array2[l];
        }
        MPI_Send ((void *)&sum_local, 1, MPI_INT, 0, 0xACE5, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
