cmake_minimum_required(VERSION 3.10)
project(cs426_project4_code C)

set(CMAKE_C_STANDARD 99)

#find_package(CUDA QUIET)


add_executable(cs426_project4_code main.c utils.c)
#cuda_add_executable( cs426_project4_code
#        jetbrainsCUDADefFix.h mmmult.cu
#)
#target_link_libraries(cs426_project4_code /usr/lib/x86_64-linux-gnu/libcudart.so.9.1)