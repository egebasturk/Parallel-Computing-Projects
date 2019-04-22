#include <stdio.h>
#include <omp.h>
#include "util.h"
#include "math.h"

/// Creates a histogram for image given by int **img and returns histogram
///  as int * hist
void create_histogram(int * hist, int ** img, int num_rows, int num_cols)
{

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
    int* histogram = malloc(FILTER_N * FILTER_N * sizeof(int));
    int** image = read_pgm_file("../images/1.1.txt", IMAGE_HEIGHT, IMAGE_WIDTH);

    create_histogram(histogram, image, FILTER_N, FILTER_N);
    return 0;
}