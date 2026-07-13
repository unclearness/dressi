# All example logic lives in dressi_examples_core (portable: desktop +
# Android). Desktop adds the GLFW live viewer + thin main() executables;
# the Android app links the core and provides its own host/viewer (JNI).
add_library(dressi_examples_core STATIC
    common/asset_utils.cpp
    common/bench.cpp
    common/geometry_utils.cpp
    common/pbr_graph.cpp
    common/example_registry.cpp
    image_fitting/run.cpp
    texture_optimization/run.cpp
    silhouette_optimization/run.cpp
    pbr_shading/run.cpp
    pbr_material_optimization/run.cpp
    pbr_envmap_optimization/run.cpp)
target_link_libraries(dressi_examples_core PUBLIC dressi dressi_assets
                                                  spdlog::spdlog)
if(MSVC)
    target_compile_options(dressi_examples_core PRIVATE /W4 /permissive-
                                                        /utf-8)
endif()

if(NOT ANDROID)
    add_library(dressi_examples_desktop STATIC
        common/vk_viewer.cpp
        common/desktop_host.cpp)
    target_link_libraries(dressi_examples_desktop
        PUBLIC dressi_examples_core dressi_glad glfw)

    foreach(_tgt image_fitting texture_optimization silhouette_optimization
            pbr_shading pbr_material_optimization pbr_envmap_optimization)
        add_executable(${_tgt} ${_tgt}/main.cpp)
        target_link_libraries(${_tgt} PRIVATE dressi_examples_desktop)
        if(MSVC)
            target_compile_options(${_tgt} PRIVATE /W4 /permissive- /utf-8)
        endif()
    endforeach()
endif()
