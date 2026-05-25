cmake_minimum_required(VERSION 3.24)
project(grit-bench-standard)

find_package(CLI11 CONFIG REQUIRED)
find_package(h5pp 1.11.0...1.11.3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME}
        cli.cpp
        hdf5_io.cpp
        main.cpp
        matrix_market.cpp
        memory.cpp
        report.cpp
        runner.cpp
        text.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE grit::grit CLI11::CLI11 h5pp::h5pp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/bench)
target_compile_definitions(${PROJECT_NAME} PRIVATE GRIT_BENCH_DATA_DIR="${CMAKE_SOURCE_DIR}/bench/data")
