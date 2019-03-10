#include <stdio.h>
#include <malloc.h>
#include <time.h>

int readArrayFromFile(int* arr[], char* input_filename)
{
    //char* input_filename = "../input_100";
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
    clock_t start, end;
    start = clock();

    int* array;
    int arr_lenght = 0;
    arr_lenght = readArrayFromFile(&array, argv[1]);
    int sum = 0;
    for (int i = 0; i < arr_lenght; ++i)
    {
        //printf("%d\n", array[i]);
        sum += array[i];
    }
    printf("%d\n", sum);
    free(array);
    end = clock();
    printf("Time Elapsed: %f", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}