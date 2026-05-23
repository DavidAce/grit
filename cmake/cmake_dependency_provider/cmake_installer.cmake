function(grit_install_one_dependency package_name source_dir)
    set(package_build_dir "${GRIT_DEPS_BUILD_DIR}/${package_name}")
    execute_process(
            COMMAND "${CMAKE_COMMAND}"
            -S "${source_dir}"
            -B "${package_build_dir}"
            -G "${CMAKE_GENERATOR}"
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${GRIT_DEPS_INSTALL_DIR}
            RESULT_VARIABLE package_configure_result)
    if(NOT package_configure_result EQUAL 0)
        message(FATAL_ERROR "Failed to configure ${package_name} CMake dependency install")
    endif()

    execute_process(
            COMMAND "${CMAKE_COMMAND}" --build "${package_build_dir}"
            RESULT_VARIABLE package_build_result)
    if(NOT package_build_result EQUAL 0)
        message(FATAL_ERROR "Failed to build/install ${package_name} CMake dependency")
    endif()
endfunction()

function(grit_install_dependencies method package_name)
    if(NOT "${method}" STREQUAL "FIND_PACKAGE")
        return()
    endif()

    if(NOT "${package_name}" STREQUAL "Eigen3" AND
       NOT "${package_name}" STREQUAL "fmt" AND
       NOT "${package_name}" STREQUAL "spdlog")
        return()
    endif()

    if(NOT DEFINED GRIT_DEPS_BUILD_DIR)
        set(GRIT_DEPS_BUILD_DIR "${CMAKE_BINARY_DIR}/pkg-build")
    endif()
    if(NOT DEFINED GRIT_DEPS_INSTALL_DIR)
        set(GRIT_DEPS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH "${GRIT_DEPS_INSTALL_DIR}")
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE INTERNAL "Paths for find_package config lookup" FORCE)

    if("${package_name}" STREQUAL "Eigen3")
        find_package(Eigen3 5 QUIET BYPASS_PROVIDER)
        if(Eigen3_FOUND)
            return()
        endif()
        grit_install_one_dependency(Eigen3 "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/external_Eigen3")
        find_package(Eigen3 ${ARGN} REQUIRED BYPASS_PROVIDER)
    elseif("${package_name}" STREQUAL "fmt")
        find_package(fmt CONFIG QUIET BYPASS_PROVIDER)
        if(fmt_FOUND)
            return()
        endif()
        grit_install_one_dependency(fmt "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/external_fmt")
        find_package(fmt CONFIG REQUIRED BYPASS_PROVIDER)
    elseif("${package_name}" STREQUAL "spdlog")
        find_package(spdlog CONFIG QUIET BYPASS_PROVIDER)
        if(spdlog_FOUND)
            return()
        endif()
        find_package(fmt CONFIG REQUIRED BYPASS_PROVIDER)
        grit_install_one_dependency(spdlog "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/external_spdlog")
        find_package(spdlog CONFIG REQUIRED BYPASS_PROVIDER)
    endif()
endfunction()
