#include <stdio.h>
#include <malloc.h>
#define LINE_LENGHT 2048
#define INPUT_ELEMENT_SIZE 2048


int readArrayFromFile(int* arr[])
{
    char* input_filename = "../test-input.txt";
    FILE *fptr;
    char *line;
    size_t len = 0;
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
int main() {
    int* array;
    int arr_lenght = 0;
    arr_lenght = readArrayFromFile(&array);

    for (int i = 0; i < arr_lenght; ++i)
    {
        printf("%d\n", array[i]);
    }
    return 0;
}