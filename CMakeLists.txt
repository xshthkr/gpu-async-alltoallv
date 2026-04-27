cmake_minimum_required (VERSION 3.10)

project (RBURKV)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(async)
add_subdirectory(examples)
add_subdirectory(benchmarks)
add_subdirectory(tests)

