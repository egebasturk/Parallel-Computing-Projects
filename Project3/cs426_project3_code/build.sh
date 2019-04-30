cc util.* lbp_seq.c -pg -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o lbp_seq -lm
cc util.* lbp_omp.c -pg -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -o lbp_omp -lm -fopenmp
