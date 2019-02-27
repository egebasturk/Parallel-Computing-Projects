#include <stdio.h>
#include <malloc.h>
#define LINE_LENGHT 2048
#define INPUT_ELEMENT_SIZE 2048


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
int main() {
    int* array1;
    int* array2;
    int* result_array;
    int N = 0;
    int arr_lenght1 = 0;
    int arr_lenght2 = 0;
    char* input_filename1 = "../test-input-matrix1.txt";
    char* input_filename2 = "../test-input-matrix2.txt";
    arr_lenght1 = readMatrixFromFile(&array1, input_filename1, &N);
    arr_lenght2 = readMatrixFromFile(&array2, input_filename2, &N);
    result_array = malloc(N * N * sizeof(int));

    printf("Matrix 1\n");
    for (int i = 0; i < arr_lenght1; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", array1[i]);
    }
    printf("\nMatrix 2\n");
    for (int i = 0; i < arr_lenght2; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", array2[i]);
    }

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



    printf("\nResult\n");
    for (int i = 0; i < N * N; ++i)
    {
        if ((i % N) == 0)
            printf("\n");
        printf("%d ", result_array[i]);
    }

    return 0;
}