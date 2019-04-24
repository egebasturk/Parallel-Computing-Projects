#include <stdio.h>
#include <omp.h>
#include "util.h"
#include "math.h"

///
u_int8_t apply_filter_on_pixel(int** img, int row, int col)
{
    u_int8_t decimal = 0;
    int center = img[row][col];
    decimal |= img[row - 1][col - 1] < center;
    decimal <<= 1;
    decimal |= img[row - 1][col] < center;
    decimal <<= 1;
    decimal |= img[row - 1][col + 1] < center;
    decimal <<= 1;
    decimal |= img[row][col + 1] < center;
    decimal <<= 1;
    decimal |= img[row + 1][col + 1] < center;
    decimal <<= 1;
    decimal |= img[row + 1][col] < center;
    decimal <<= 1;
    decimal |= img[row + 1][col - 1] < center;
    decimal <<= 1;
    decimal |= img[row][col - 1] < center;
    return decimal;
}

/// Creates a histogram for image given by int **img and returns histogram
///  as int * hist
void create_histogram(int * hist, int ** img, int num_rows, int num_cols)
{
    u_int8_t** img_lgb = alloc_2d_matrix_unsigned(num_rows, num_cols);
    if (img_lgb == NULL)
        printf("NULL Pointer\n");
    for (int i = 1; i < IMAGE_HEIGHT - 2; ++i) {
        for (int j = 1; j < IMAGE_WIDTH - 2; ++j) {
            img_lgb[i][j] = apply_filter_on_pixel(img, i, j);
        }
    }
    // Pixels will fit, ignore warning
    for (int k = 0; k < IMAGE_HEIGHT; ++k) {
        img_lgb[k][0] = img[k][0];
        img_lgb[k][IMAGE_WIDTH - 1] = img[k][IMAGE_WIDTH - 1];
    }
    for (int l = 0; l < IMAGE_WIDTH; ++l) {
        img_lgb[0][l] = img[0][l];
        img_lgb[IMAGE_HEIGHT - 1][l] = img[IMAGE_HEIGHT - 1][l];
    }

    dealloc_2d_matrix((int**)img_lgb, num_rows, num_cols); // Dangerous cast but should work with free
}

/// Finds the distance between two vectors
double distance(int * a, int *b, int size)
{
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

/// Finds the closest histogram for test image's histogram from training set
/// histograms
/// Returns person id of the closest histogram
int find_closest(int ***training_set, int num_persons, int num_training, int size,
                 int * test_image)
{

}
int main()
{
    int* histogram = malloc((IMAGE_HEIGHT - 2) * (IMAGE_WIDTH - 2) * sizeof(int));
    int** image = read_pgm_file("../images/1.1.txt", IMAGE_HEIGHT, IMAGE_WIDTH);

    create_histogram(histogram, image, IMAGE_HEIGHT, IMAGE_WIDTH);

    dealloc_2d_matrix(image, IMAGE_HEIGHT, IMAGE_WIDTH);
    free(histogram);
    return 0;
}