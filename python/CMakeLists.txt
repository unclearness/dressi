# dressi Python bindings (nanobind). Enabled by DRESSI_BUILD_PYTHON;
# point Python_EXECUTABLE at the target interpreter (e.g. the project venv).
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.7.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(dressi_py NB_STATIC csrc/module.cpp)
set_target_properties(dressi_py PROPERTIES OUTPUT_NAME "_C")
target_link_libraries(dressi_py PRIVATE dressi)
if(MSVC)
    target_compile_options(dressi_py PRIVATE /bigobj /utf-8)
endif()

# Drop the module next to the package sources so PYTHONPATH=python works
add_custom_command(TARGET dressi_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:dressi_py>
            ${CMAKE_CURRENT_SOURCE_DIR}/dressi/$<TARGET_FILE_NAME:dressi_py>)

if(DRESSI_BUILD_TESTS)
    add_test(NAME python_bindings
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/python -q)
    set_tests_properties(python_bindings PROPERTIES
        LABELS gpu
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
endif()
