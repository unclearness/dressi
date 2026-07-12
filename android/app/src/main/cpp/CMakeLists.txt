cmake_minimum_required(VERSION 3.28)
project(dressi_android CXX C)

# Library-only pull of the repo root: no tests, no configure-time dataset
# download (the app bundles a curated asset subset via a Gradle sync task).
set(DRESSI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DRESSI_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(DRESSI_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(DRESSI_FETCH_DATA OFF CACHE BOOL "" FORCE)

get_filename_component(DRESSI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
add_subdirectory("${DRESSI_ROOT}" dressi_root)

add_library(dressi_examples SHARED
    jni_bridge.cpp
    android_host.cpp
    device_caps.cpp)
target_include_directories(dressi_examples PRIVATE
    "${DRESSI_ROOT}/examples/common")
target_link_libraries(dressi_examples PRIVATE
    dressi_examples_core
    android
    log
    nativewindow)
