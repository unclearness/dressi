# Writes OUTPUT defining DRESSI_GIT_COMMIT: the short commit of SOURCE_DIR, with
# -dirty if the tree has modifications. Rewrites only on change, so an unchanged
# commit does not recompile anything. Run with cmake -P.
find_package(Git QUIET)
set(_commit "unknown")
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}"
                            rev-parse --short=12 HEAD
                    OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _head_rc ERROR_QUIET)
    if(_head_rc EQUAL 0)
        set(_commit "${_head}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}"
                                status --porcelain
                        OUTPUT_VARIABLE _status ERROR_QUIET)
        if(NOT _status STREQUAL "")
            string(APPEND _commit "-dirty")
        endif()
    endif()
endif()
set(_content "#define DRESSI_GIT_COMMIT \"${_commit}\"\n")
set(_old "")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" _old)
endif()
if(NOT _old STREQUAL _content)
    file(WRITE "${OUTPUT}" "${_content}")
endif()
