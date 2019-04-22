cmake_minimum_required(VERSION 3.10)
project(cs426_project3_code C)

set(CMAKE_C_STANDARD 99)

add_executable(cs426_project3_code lbp_seq.c util.c util.h)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(cs426_project3_code PUBLIC OpenMP::OpenMP_C m)
endif()