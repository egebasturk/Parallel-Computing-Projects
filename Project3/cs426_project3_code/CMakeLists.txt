cmake_minimum_required(VERSION 3.10)
project(cs426_project3_code C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -O0")
# -fsanitize-recover=address
add_executable(cs426_project3_code lbp_seq.c util.c util.h )
add_executable(IO_test IO_test.c util.c util.h )

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(cs426_project3_code PUBLIC OpenMP::OpenMP_C m)
endif()