cmake_minimum_required(VERSION 3.10)
project(cs426_project3_code C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg -fsanitize=address -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls")
# -pg -fsanitize=address -fsanitize-recover=address
# optimizations -O2 -fno-omit-frame-pointer -fno-optimize-sibling-calls
add_executable(lbp_seq lbp_seq.c util.c util.h )
add_executable(lbp_omp lbp_omp.c util.c util.h )
add_executable(IO_test IO_test.c util.c util.h )

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(lbp_omp PUBLIC OpenMP::OpenMP_C m)
    target_link_libraries(lbp_seq PUBLIC OpenMP::OpenMP_C m)
endif()