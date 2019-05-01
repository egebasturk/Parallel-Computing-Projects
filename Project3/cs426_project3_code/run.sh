#!/bin/bash
#cd cmake-build-debug
# initial tests are used to generate prof_omp and prof_sequential texts
echo "----INITIAL TEST----(k = 10, 4 threads for omp)"
export OMP_NUM_THREADS=4
./lbp_omp 10 # > alpege_basturk.output
#gprof ./lbp_omp | gprof2dot -n0 -e0 | dot -Tpng -o omp-png.png # Enable to generate png (needs gprof2dot.py)
#gprof ./lbp_omp gmon.out > prof_omp.txt
./lbp_seq 10 # >> alpege_basturk.output
#gprof ./lbp_seq | gprof2dot -n0 -e0 | dot -Tpng -o seq-png.png # Enable to generate png (needs gprof2dot.py)
#gprof ./lbp_seq gmon.out > prof_sequential.txt

# k values
# 1 2 5 7 10
# run for different thread numbers
#1 2 4 6 8 16
echo "----TESTS WITH DIFFERENT PARAMETERS----"
threadsArray=(1 2 4 6 8 16)
kArray=(1 2 5 7 10)
for k in "${kArray[@]}"; do
	echo "Sequential running with k = $k"
	if (($k == 1)); then
	    ./lbp_seq 1 > alpege_basturk.output # writing first one like this to overwrite anything left
	    gprof ./lbp_seq gmon.out > prof_sequential.txt
	else
	    ./lbp_seq $k >> alpege_basturk.output
	    gprof ./lbp_seq gmon.out >> prof_sequential.txt
    fi

	for t in "${threadsArray[@]}"; do
        echo "OMP running with $t threads, where k = $k"
		export OMP_NUM_THREADS=$t
	    ./lbp_omp $k >> alpege_basturk.output

		if (($k == 1)); then
            gprof ./lbp_omp gmon.out > prof_omp.txt
        else
            gprof ./lbp_omp gmon.out >> prof_omp.txt
        fi
	done
done
