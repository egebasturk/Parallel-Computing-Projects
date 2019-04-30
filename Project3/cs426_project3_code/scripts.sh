cd cmake-build-debug
./lbp_omp 10
gprof ./lbp_omp | gprof2dot -n0 -e0 | dot -Tpng -o omp-png.png
gprof ./lbp_omp gmon.out > analysis-omp.txt
./lbp_seq 10
gprof ./lbp_seq | gprof2dot -n0 -e0 | dot -Tpng -o seq-png.png
gprof ./lbp_seq gmon.out > analysis-seq.txt
