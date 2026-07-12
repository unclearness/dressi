# Desktop-only helper tools (not built for Android).
add_executable(downsample_exr downsample_exr/main.cpp)
target_link_libraries(downsample_exr PRIVATE dressi_examples_core)
if(MSVC)
    target_compile_options(downsample_exr PRIVATE /W4 /permissive- /utf-8)
endif()
