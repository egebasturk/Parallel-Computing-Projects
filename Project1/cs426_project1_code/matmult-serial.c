#include <stdio.h>
#include <malloc.h>

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
    printf("Converted");
    return return_array;
}
int main (int argc, char *argv[]) {
    int* array1;
    int* array2;
    int* result_array;
    int N = 0;
    int arr_lenght1 = 0;
    int arr_lenght2 = 0;
    char* input_filename1 = argv[1];
    char* input_filename2 = argv[2];
    char* output_file_name = argv[3];
    arr_lenght1 = readMatrixFromFile(&array1, input_filename1, &N);
    arr_lenght2 = readMatrixFromFile(&array2, input_filename2, &N);
    result_array = malloc(N * N * sizeof(int));

    /*printf("Matrix 1\n");
    for (int i = 0; i < arr_lenght1; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", array1[i]);
    }*/
    /*printf("\nMatrix 2\n");
    for (int i = 0; i < arr_lenght2; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", array2[i]);
    }*/

    // 1D matrix matrix mult
    int sum_tmp = 0;
    for (int row = 0; row < N; ++row) {
        for (int column = 0; column < N; ++column) {
            for (int i = 0; i < N; ++i) {
                sum_tmp += array1[row * N + i] * array2[column + i * N];
            }
            result_array[column + row * N] = sum_tmp;
            sum_tmp = 0;
        }
    }

    //printf("\nResult\n");
    FILE *fptr;
    fptr = fopen(output_file_name, "w");
    fprintf(fptr, "%d\n", N);
    for (int i = 0; i < N * N; ++i)
    {
        if ((i % N) == 0 && i != 0)
            fprintf(fptr, "\n");
        fprintf(fptr, "%d ", result_array[i]);
    }

    /*
     * Test code for matrix col wise conversion
    int* array2_colwise = convertRowWiseMatrixToColumnWise(array2, N);
    printf("\ncol wise conversion\n");
    for (int i = 0; i < N * N; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", array2_colwise[i]);
    }*/
    free(array1);
    free(array2);
    return 0;
}