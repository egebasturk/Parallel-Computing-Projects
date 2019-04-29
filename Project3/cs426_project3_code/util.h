#include <stdlib.h>
#include <string.h>
#include <stdio.h>


/*
*	Allocates an rxc integer matrix
*/
int ** alloc_2d_matrix(int r, int c);

/*
*	Deallocates an rxc integer matrix
*/
void dealloc_2d_matrix(int ** a, int r, int c);

/*@params: 
*		file_name: name of the file
*		h: number of rows in the file
*		w: number of columns in the file
*		reads a matrix file
*		note that this function can not read pgm files, only use with given dataset
**/ 
int ** read_pgm_file(char * file_name, int h, int w);

// My helpers
#include <limits.h>
#include "omp.h"
#include "math.h"
#include <time.h>
#include <sys/time.h>
#define IMAGE_HEIGHT 200
#define IMAGE_WIDTH 180
#define DEBUG_IMG_WRITE 0
#define DEBUG_LBP_WRITE 0
#define FILTER_N 3
u_int8_t** alloc_2d_matrix_unsigned(int r, int c);