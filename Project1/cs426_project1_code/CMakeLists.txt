cmake_minimum_required(VERSION 3.10)
project(cs426_project1_code C)

set(CMAKE_C_STANDARD 99)

#add_executable(cs426_project1_code sum-serial.c)
add_executable(serial1 sum-serial.c)

include_directories(/usr/include/mpi/)
link_libraries(mpi)
add_executable(mpi1 sum-mpi-ppv1.c)