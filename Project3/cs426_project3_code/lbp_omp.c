#include <stdio.h>
#include "util.h"
#define INVALID_NUM_INIT_VALUE (-666)
#define GET_TIME omp_get_wtime()

/// Apply filter on single pixel location
u_int8_t apply_filter_on_pixel(int** img, int row, int col)
{
    u_int8_t decimal = 0;
    int center = img[row][col];
    /// 1st row
    decimal |= img[row - 1][col - 1] > center;
    decimal = decimal << 1u;
    decimal |= img[row - 1][col]     > center;
    decimal = decimal << 1u;
    decimal |= img[row - 1][col + 1] > center;
    decimal = decimal << 1u;
    /// 2nd col
    decimal |= img[row][col + 1]     > center;
    decimal = decimal << 1u;
    /// 3rd row
    decimal |= img[row + 1][col + 1] > center;
    decimal = decimal << 1u;
    decimal |= img[row + 1][col]     > center;
    decimal = decimal << 1u;
    decimal |= img[row + 1][col - 1] > center;
    decimal = decimal << 1u;
    /// 2nd row again
    decimal |= img[row][col - 1]     > center;
    return decimal;
}

/// Creates a histogram for image given by int **img and returns histogram
///  as int * hist
int flag = 15;
void create_histogram(int * hist, int ** img, int num_rows, int num_cols)
{
    int** img_lbp;
    //if (DEBUG_LBP_WRITE) {
    #if DEBUG_LBP_WRITE
    img_lbp = alloc_2d_matrix(num_rows, num_cols);
    printf("PARALLEL_DEBUGDEBUGDEBUGDEBUG\n");
    if (img_lbp == NULL)
        printf("NULL Pointer\n");
    #endif
    //}
    for (int i = 1; i < IMAGE_HEIGHT - 1; ++i) {
        for (int j = 1; j < IMAGE_WIDTH - 1; ++j) {
            int tmp = apply_filter_on_pixel(img, i, j);
            //if (DEBUG_LBP_WRITE)
            #if DEBUG_LBP_WRITE
                img_lbp[i][j] = tmp;
            #endif
            ((int*)hist)[tmp]++;
        }
    }
    //if (DEBUG_LBP_WRITE) {
    #if DEBUG_LBP_WRITE
    // Pixels will fit, ignore warning
    for (int k = 0; k < IMAGE_HEIGHT; ++k) {
        img_lbp[k][0] = 0;// img[k][0];
        img_lbp[k][IMAGE_WIDTH - 1] = 0;// img[k][IMAGE_WIDTH - 1];
        //((int*)hist)[0] += 2;
    }
    for (int l = 0; l < IMAGE_WIDTH; ++l) {
        img_lbp[0][l] = 0;// img[0][l];
        img_lbp[IMAGE_HEIGHT - 1][l] = 0;//img[IMAGE_HEIGHT - 1][l];
        //((int*)hist)[0] += 2;
    }
    if (DEBUG_IMG_WRITE & flag == 15) {
        FILE *fptr;
        if (fptr = fopen("mat.out", "w")) {
            for (int i = 0; i < IMAGE_HEIGHT; ++i) {
                for (int j = 0; j < IMAGE_WIDTH; ++j) {
                    fprintf(fptr, "%d ", img_lbp[i][j]);
                }
                fprintf(fptr, "\n");
            }
        } else
            printf("ADGFXNCH\n");
        flag = 1;
    }
    dealloc_2d_matrix((int **) img_lbp, num_rows, num_cols); // Dangerous cast but should work with free
    //}
    #endif
}

/// Finds the distance between two vectors
double distance(int * a, int *b, int size)
{
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        if ((double)(a[i] + b[i]) != 0)
            sum += pow(a[i] - b[i], 2) / (double)(a[i] + b[i]);
    }
    return sum / 2;
}

/// Finds the closest histogram for test image's histogram from training set
/// histograms
/// Returns person id of the closest histogram
int find_closest(int ***training_set, int num_persons, int num_training, int size,
                 int * test_image)
{
    double min = INT_MAX;
    int min_id_i = INVALID_NUM_INIT_VALUE;
    int min_id_j;
    for (int i = 0; i < num_persons; ++i) {
        for (int j = 0; j < num_training; ++j) {
            double tmp = distance(training_set[i][j], test_image, size);
            if (tmp < min) {
                min = tmp;
                min_id_i = i;
                min_id_j = j;
            }
        }
    }
    return min_id_i;
}
int main(int argc, char* argv[])
{
    double start = GET_TIME;
    int k = atoi(argv[1]), people_count = 18, sample_count_per_person = 20;
    char* buff = malloc(32 * sizeof(char));
    int**** original_images = malloc(people_count * sizeof(int***));

    /// Create Arrays for each person, 18 arrays in this case
    int*** histogram_array = malloc(people_count * sizeof(int**));
    /// Create histograms inside portion for each person, 20 for each person

    #pragma omp for
    for (int i = 0; i < people_count; ++i) {
        histogram_array[i] = malloc(sample_count_per_person * sizeof(int *));
        original_images[i] = malloc(sample_count_per_person * sizeof(int **));
    }
    #pragma omp for collapse(2)
    for (int i = 0; i < people_count; ++i) {
        for (int j = 0; j < sample_count_per_person; ++j) {
            sprintf(buff, "../images/%d.%d.txt", i + 1, j + 1); // Arrays don't start from zero
            int **image = read_pgm_file(buff, IMAGE_HEIGHT, IMAGE_WIDTH);
            original_images[i][j] = image;
        }
    }

    #pragma omp for
    for (int i = 0; i < people_count; ++i)
    {
        //histogram_array[i] = malloc(sample_count_per_person * sizeof(int*));
        //original_images[i] = malloc(sample_count_per_person * sizeof(int**));
        for (int j = 0; j < sample_count_per_person; ++j)
        {
            //sprintf(buff, "../images/%d.%d.txt", i + 1, j + 1); // Arrays don't start from zero
            //int** image = read_pgm_file(buff, IMAGE_HEIGHT, IMAGE_WIDTH);
            //original_images[i][j] = image;

            histogram_array[i][j] = calloc(256, sizeof(int));
            create_histogram((int*)histogram_array[i][j], original_images[i][j], IMAGE_HEIGHT, IMAGE_WIDTH);//Dangerous cast

            //dealloc_2d_matrix(image, IMAGE_HEIGHT, IMAGE_WIDTH);
        }
    }
    /// Test
    int correct_count = 0;
    int incorrect_count = 0;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < people_count; ++i)
    {
        for (int j = k; j < sample_count_per_person; ++j)
        {
            sprintf(buff, "%d.%d.txt", i + 1, j + 1); // Arrays don't start from zero so add 1
            int found_person_id = find_closest(histogram_array, people_count, k, 256, histogram_array[i][j]);
            //printf("%d\n", found_person_id);

            if (found_person_id == i)
                correct_count++;
            else
                incorrect_count++;

            /// Print intermediate results as asked
            printf("%s %d %d\n", buff, found_person_id + 1, i + 1);
        }
    }
    /// Print all results
    printf("Accuracy: %d correct answers for %d tests\n", correct_count,
           people_count * sample_count_per_person - k * people_count);


    /// Cleanup
    for (int l = 0; l < people_count; ++l) {
        for (int i = 0; i < sample_count_per_person; ++i) {
            free(histogram_array[l][i]);
            dealloc_2d_matrix(original_images[l][i], IMAGE_HEIGHT, IMAGE_WIDTH);
        }
        free(histogram_array[l]);
        free(original_images[l]);
    }
    free(histogram_array);
    free(original_images);
    free(buff);
    double end = GET_TIME;
    printf("Parallel Time: %lf\n", end - start);
    return 0;
}