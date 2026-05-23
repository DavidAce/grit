cmake_minimum_required(VERSION 3.24)
project(grit_consumer LANGUAGES CXX)

find_package(grit REQUIRED)

add_executable(grit-consumer main.cpp)
target_link_libraries(grit-consumer PRIVATE grit::grit)
