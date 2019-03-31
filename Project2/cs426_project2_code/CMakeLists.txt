cmake_minimum_required(VERSION 3.10)
project(cs426_project2_code C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -O0")

include_directories(/usr/include/mpi/)
link_libraries(mpi)
link_libraries(m)

#add_executable(cs426_project2_code main.c)
add_executable(main main.c utils.h utils.c)