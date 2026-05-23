cmake_minimum_required(VERSION 3.24)
project(grit
        VERSION 0.1.0
        DESCRIPTION "Generalized Ritz Iteration Toolkit"
        HOMEPAGE_URL "https://github.com/DavidAce/grit"
        LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

option(GRIT_ENABLE_TESTS "Enable unit testing with ctest" OFF)
option(GRIT_BUILD_EXAMPLES "Build example programs" OFF)
option(GRIT_CMAKE_DEBUG "Print extra CMake configuration details" OFF)

option(COMPILER_PROFILE_BUILD "Enable -ftime-trace where supported" OFF)
option(COMPILER_ENABLE_ASAN "Enable expensive runtime sanitizers" OFF)
option(COMPILER_ENABLE_USAN "Enable cheap runtime sanitizers" OFF)
option(COMPILER_ENABLE_PCH "Enable precompiled headers" OFF)
option(COMPILER_ENABLE_CCACHE "Enable ccache compiler launcher when available" OFF)
option(COMPILER_ENABLE_COVERAGE "Enable test coverage" OFF)
set(COMPILER_EXTRA_WARNINGS "" CACHE STRING "Extra C++ warning flags")

set(GRIT_PACKAGE_MANAGERS_VALID find cmake conan)
set(GRIT_PACKAGE_MANAGER find CACHE STRING "Package manager for external dependencies")
set_property(CACHE GRIT_PACKAGE_MANAGER PROPERTY STRINGS ${GRIT_PACKAGE_MANAGERS_VALID})
if(NOT GRIT_PACKAGE_MANAGER IN_LIST GRIT_PACKAGE_MANAGERS_VALID)
    message(FATAL_ERROR "GRIT_PACKAGE_MANAGER must be one of ${GRIT_PACKAGE_MANAGERS_VALID}")
endif()

add_library(grit-config INTERFACE)

include(cmake/SetupCcache.cmake)
include(cmake/SetupScalars.cmake)
include(cmake/CompilerFlags.cmake)
include(cmake/SetupPaths.cmake)
include(cmake/SetupDependencies.cmake)
include(cmake/GenerateInstantiationSources.cmake)

generate_scalar_instantiations(grit_instantiation_sources
                               TEMPLATE source/instantiations/grit.inst.cpp.in
                               OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/instantiations")

add_library(grit-objects OBJECT ${grit_instantiation_sources})
target_include_directories(grit-objects PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(grit-objects PUBLIC grit-config)
target_link_precompiled_headers(grit-objects)

add_library(grit)
add_library(grit::grit ALIAS grit)
target_sources(grit PRIVATE $<TARGET_OBJECTS:grit-objects>)
target_include_directories(grit
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(grit PUBLIC cxx_std_23)
if(GRIT_PUBLIC_ARCH_OPTIONS)
    target_compile_options(grit PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${GRIT_PUBLIC_ARCH_OPTIONS}>)
endif()
grit_apply_scalar_config(grit)
target_link_libraries(grit PUBLIC ${GRIT_PUBLIC_DEPENDENCIES})
target_enable_static_libgcc(grit)

set_target_properties(grit PROPERTIES
                      VERSION ${PROJECT_VERSION}
                      SOVERSION ${PROJECT_VERSION_MAJOR}
                      EXPORT_NAME grit)

install(TARGETS grit
        EXPORT gritTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/grit DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/gritConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/gritConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/gritConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grit")

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/gritConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/gritConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grit")

install(DIRECTORY cmake/modules DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grit")

install(EXPORT gritTargets
        NAMESPACE grit::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/grit")

if(GRIT_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GRIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GRIT_CMAKE_DEBUG)
    message(STATUS "grit package manager: ${GRIT_PACKAGE_MANAGER}")
    message(STATUS "grit install prefix: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "grit enabled scalars: ${GRIT_ENABLED_SCALARS}")
endif()
