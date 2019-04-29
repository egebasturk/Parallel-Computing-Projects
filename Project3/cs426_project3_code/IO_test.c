//
// Created by egebasturk on 4/28/19.
//
#include "util.h"
int flag = 15;
int main()
{
    if (flag == 15)
    {
        int** image = read_pgm_file("../images/1.1.txt", IMAGE_HEIGHT, IMAGE_WIDTH);
        FILE* fptr;
        if (fptr = fopen("mat.out", "w"))
        {
            for (int i = 0; i < IMAGE_HEIGHT; ++i) {
                for (int j = 0; j < IMAGE_WIDTH; ++j) {
                    //fwrite((void*)&image[i][j], sizeof(int), 1, fptr);
                    fprintf(fptr, "%lu ", image[i][j]);
                }
                fprintf(fptr, "\n");
            }
        } else
            printf("ADGFXNCH\n");
        flag = 1;

        dealloc_2d_matrix(image, IMAGE_HEIGHT, IMAGE_WIDTH);
    }
}