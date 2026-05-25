unset(PKG_IS_RUNNING CACHE) # Remove from cache when this file is included

function(grit_install_dependencies method package_name)
    if(NOT PKG_INSTALL_SUCCESS AND NOT PKG_IS_RUNNING)
        message(STATUS "grit_install_dependencies running with package_name: ${package_name}")
        unset(PKG_INSTALL_SUCCESS CACHE)
        set(PKG_IS_RUNNING TRUE CACHE INTERNAL "" FORCE)
        include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PKGInstall.cmake)

        pkg_install(Eigen3)
        pkg_install(fmt)
        pkg_install(spdlog)

        if(GRIT_ENABLE_BENCH)
            pkg_install(cli11)
            pkg_install(h5pp)
        endif()

        set(PKG_INSTALL_SUCCESS TRUE CACHE BOOL "PKG dependency install has been invoked and was successful")
        set(PKG_IS_RUNNING FALSE CACHE INTERNAL "" FORCE)
    endif()

    find_package(${ARGN} BYPASS_PROVIDER)
endfunction()
