# All example logic lives in dressi_examples_core (portable: desktop +
# Android + iOS). Desktop adds the GLFW live viewer + thin main() executables;
# the mobile apps link the core and provide their own host/viewer.
add_library(dressi_examples_core STATIC
    common/asset_utils.cpp
    common/bench.cpp
    common/geometry_utils.cpp
    common/pbr_graph.cpp
    common/example_registry.cpp
    image_fitting/run.cpp
    texture_optimization/run.cpp
    silhouette_optimization/run.cpp
    shape_texture_optimization/run.cpp
    pbr_shading/run.cpp
    pbr_material_optimization/run.cpp
    pbr_envmap_optimization/run.cpp)
target_link_libraries(dressi_examples_core PUBLIC dressi dressi_assets
                                                  spdlog::spdlog)
if(MSVC)
    target_compile_options(dressi_examples_core PRIVATE /W4 /permissive-
                                                        /utf-8)
endif()

# bench.json records the build. The commit is regenerated on every build, not
# at configure time, so a record cannot name a stale commit.
set(_git_commit_h "${CMAKE_CURRENT_BINARY_DIR}/generated/dressi_git_commit.h")
add_custom_target(dressi_git_commit
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DOUTPUT=${_git_commit_h}
            -P ${PROJECT_SOURCE_DIR}/cmake/GitCommit.cmake
    BYPRODUCTS "${_git_commit_h}"
    VERBATIM)
add_dependencies(dressi_examples_core dressi_git_commit)
target_include_directories(dressi_examples_core PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(dressi_examples_core PRIVATE
    DRESSI_BUILD_TYPE="$<CONFIG>")

if(NOT DRESSI_MOBILE)
    add_library(dressi_examples_desktop STATIC
        common/vk_viewer.cpp
        common/desktop_host.cpp)
    target_link_libraries(dressi_examples_desktop
        PUBLIC dressi_examples_core dressi_glad glfw)

    foreach(_tgt image_fitting texture_optimization silhouette_optimization
            shape_texture_optimization
            pbr_shading pbr_material_optimization pbr_envmap_optimization)
        add_executable(${_tgt} ${_tgt}/main.cpp)
        target_link_libraries(${_tgt} PRIVATE dressi_examples_desktop)
        if(MSVC)
            target_compile_options(${_tgt} PRIVATE /W4 /permissive- /utf-8)
        endif()
    endforeach()
endif()
