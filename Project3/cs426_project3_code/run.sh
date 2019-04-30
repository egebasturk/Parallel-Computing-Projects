#!/bin/bash
#cd cmake-build-debug
echo "----INITIAL TEST----(k = 10, 4 threads for omp)"
./lbp_omp 10 4 # > alpege_basturk.output
gprof ./lbp_omp | gprof2dot -n0 -e0 | dot -Tpng -o omp-png.png
gprof ./lbp_omp gmon.out > prof_omp.txt
./lbp_seq 10 # >> alpege_basturk.output
gprof ./lbp_seq | gprof2dot -n0 -e0 | dot -Tpng -o seq-png.png
gprof ./lbp_seq gmon.out > prof_sequential.txt

# k values
# 1 2 5 7 10
# run for different thread numbers
#1 2 4 6 8 16
echo "----TESTS WITH DIFFERENT PARAMETERS----"
threadsArray=(1 2 4 6 8 16)
kArray=(1 2 5 7 10)
for k in "${kArray[@]}"; do
	echo "Sequential running with k = $k"
	./lbp_seq $k >> alpege_basturk.output
done
for t in "${threadsArray[@]}"; do
    echo "OMP running for $t threads, where k = 15"
    ./lbp_omp 15 $t >> alpege_basturk.output
done
