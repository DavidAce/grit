cmake_minimum_required(VERSION 3.24)
project(examples)

file(GLOB EXAMPLES "*.cpp")
add_custom_target(build-all-examples)

foreach(ex ${EXAMPLES})
    get_filename_component(example_src ${ex} NAME)
    get_filename_component(example_nwe ${ex} NAME_WE)
    add_executable(${example_nwe} ${example_src})
    target_link_libraries(${example_nwe} PRIVATE grit::grit)
    add_dependencies(build-all-examples ${example_nwe})
endforeach()
