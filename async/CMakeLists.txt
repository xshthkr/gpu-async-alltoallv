cmake_minimum_required(VERSION 4.3)
project(async-overlap LANGUAGES CXX VERSION 1.0.0)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HWLOC REQUIRED hwloc)

add_library(async-overlap)

target_sources(async-overlap
    PRIVATE
    parlogna.cpp
    parlinna.cpp
    parlinna_servlet.cpp
    parlinna_servlet_v2.cpp
    comm_servlet.cpp
    utils.cpp
    
    PUBLIC
    FILE_SET HEADERS
    FILES
    async.h
    comm_servlet.h
)

target_include_directories(async-overlap 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${HWLOC_INCLUDE_DIRS}
)

target_link_libraries(async-overlap
    PUBLIC
    MPI::MPI_CXX
    Threads::Threads
    PRIVATE
    ${HWLOC_LIBRARIES}
)