cmake_minimum_required(VERSION 3.28)

project(opendressi CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# This project uses no C++20 modules. With Ninja + GCC/Clang, CMake would
# otherwise scan every C++20 translation unit for module dependencies
# (-fmodules-ts), which is pure overhead here and best avoided.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(DRESSI_BUILD_TESTS "Build unit tests" ON)
option(DRESSI_BUILD_EXAMPLES "Build examples" ON)
option(DRESSI_BUILD_PYTHON "Build the Python (torch) bindings" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindVulkanSdk)

find_package(Vulkan REQUIRED)

# glslang from the Vulkan SDK (runtime GLSL -> SPIR-V compilation for vkw).
# The SDK's glslang-config.cmake is broken (references a nonexistent
# <SDK>/Lib/lib/... path), so link the static libs directly.
add_library(dressi_glslang INTERFACE)
set(_glslang_libs
    glslang SPIRV MachineIndependent GenericCodeGen OSDependent
    glslang-default-resource-limits SPIRV-Tools-opt SPIRV-Tools)
if(WIN32)
    # Windows SDK nests glslang's whole tree under Include/glslang, so that one
    # include dir resolves both <SPIRV/...> and <glslang/Public/...>. Libs are
    # <name>.lib / <name>d.lib with per-config debug ("d" suffix) selection.
    target_include_directories(dressi_glslang SYSTEM INTERFACE
        "${DRESSI_VULKAN_SDK_INCLUDE}/glslang")
    foreach(_lib IN LISTS _glslang_libs)
        target_link_libraries(dressi_glslang INTERFACE
            "$<IF:$<CONFIG:Debug>,${DRESSI_VULKAN_SDK_LIB}/${_lib}d.lib,${DRESSI_VULKAN_SDK_LIB}/${_lib}.lib>")
    endforeach()
else()
    # Linux SDK flattens glslang's headers directly under include/glslang, so
    # <SPIRV/...> resolves under include/glslang while <glslang/Public/...>
    # resolves under include; add both. The archives are static lib<name>.a
    # with circular interdependencies (glslang <-> MachineIndependent <-> SPIRV),
    # so wrap them in a link group for GNU ld to resolve regardless of order.
    target_include_directories(dressi_glslang SYSTEM INTERFACE
        "${DRESSI_VULKAN_SDK_INCLUDE}"
        "${DRESSI_VULKAN_SDK_INCLUDE}/glslang")
    set(_glslang_paths "")
    foreach(_lib IN LISTS _glslang_libs)
        list(APPEND _glslang_paths "${DRESSI_VULKAN_SDK_LIB}/lib${_lib}.a")
    endforeach()
    target_link_libraries(dressi_glslang INTERFACE
        -Wl,--start-group ${_glslang_paths} -Wl,--end-group)
endif()

include(FetchContent)

# GLFW: vkw.h includes <GLFW/glfw3.h> on desktop even for headless use
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
if(UNIX AND NOT APPLE)
    # Build the X11 backend only. Wayland would additionally require
    # wayland-scanner + libwayland-dev + libxkbcommon-dev; X11-only GLFW still
    # runs under Wayland compositors via XWayland, so this keeps the build
    # dependency-light. Override with -DGLFW_BUILD_WAYLAND=ON if desired.
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw)

# Asset loaders for examples/tests (header-only usage)
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
FetchContent_MakeAvailable(tinyobjloader)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
FetchContent_MakeAvailable(stb)
# tinygltf (glTF 2.0; the successor of tinygltfloader)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.5
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
FetchContent_MakeAvailable(tinygltf)
# tinyexr (EXR env maps for the PBS examples; zlib comes from the stb
# implementations already compiled in asset_utils.cpp)
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG v1.0.9
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
FetchContent_MakeAvailable(tinyexr)
add_library(dressi_assets INTERFACE)
target_include_directories(dressi_assets SYSTEM INTERFACE
    "${tinyobjloader_SOURCE_DIR}" "${stb_SOURCE_DIR}"
    "${tinygltf_SOURCE_DIR}" "${tinyexr_SOURCE_DIR}")

# glad Vulkan loader (user-provided) for the example live viewer.
# The SDK include path comes first so the full vk_video header set is used
# (the glad drop bundles only a subset).
add_library(dressi_glad STATIC third_party/glad/src/vulkan.c)
target_include_directories(dressi_glad SYSTEM PUBLIC
    "${DRESSI_VULKAN_SDK_INCLUDE}"
    third_party/glad/include)

# Logging / performance measurement
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spdlog)

# ------------------------------------------------------------------------------
# dressi_vkw: takiyu/VulkanWrapper compiled by our own target (its bundled
# CMake/third_party predate CMake 4.x and current glslang, so we do not
# add_subdirectory it; we link against the Vulkan SDK instead).
# ------------------------------------------------------------------------------
add_library(dressi_vkw STATIC
    third_party/VulkanWrapper/src/vkw.cpp)
target_include_directories(dressi_vkw SYSTEM PUBLIC
    third_party/VulkanWrapper/include)
# Dynamic dispatch (upstream vkw's mode): debug-utils/report extension
# functions are unresolvable with the static dispatcher + vulkan-1.lib.
target_compile_definitions(dressi_vkw PUBLIC
    VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    ENABLE_OPT=1)
if(WIN32)
    target_compile_definitions(dressi_vkw PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_link_libraries(dressi_vkw PUBLIC
    Vulkan::Vulkan
    dressi_glslang
    glfw)
if(NOT WIN32)
    # The glslang/SPIRV-Tools static archives pull in pthread and dlopen.
    find_package(Threads REQUIRED)
    target_link_libraries(dressi_vkw PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

# ------------------------------------------------------------------------------
# dressi: the Dressi-AD library
# ------------------------------------------------------------------------------
file(GLOB_RECURSE DRESSI_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(dressi STATIC ${DRESSI_SOURCES})
target_include_directories(dressi PUBLIC include PRIVATE src)
target_link_libraries(dressi PUBLIC dressi_vkw PRIVATE spdlog::spdlog)
if(MSVC)
    target_compile_options(dressi PRIVATE /W4 /permissive- /bigobj /utf-8)
endif()

if(MSVC)
    target_compile_options(dressi_vkw PRIVATE /bigobj)
endif()

# Provision example / test datasets into data/ (large, .gitignore'd). Only
# needed when something consumes them; skipped for a library-only build.
if(DRESSI_BUILD_TESTS OR DRESSI_BUILD_EXAMPLES)
    include(FetchAssets)
endif()

if(DRESSI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DRESSI_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

if(DRESSI_BUILD_PYTHON)
    add_subdirectory(python)
endif()
