cmake_minimum_required(VERSION 3.24)
project(external-fmt)

include(ExternalProject)
ExternalProject_Add(external_fmt
        URL https://github.com/fmtlib/fmt/archive/refs/tags/12.0.0.tar.gz
        PREFIX ${CMAKE_BINARY_DIR}
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_UPDATE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_MESSAGE=NEVER
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DFMT_DOC:BOOL=OFF
        -DFMT_TEST:BOOL=OFF
        -DFMT_INSTALL:BOOL=ON)
