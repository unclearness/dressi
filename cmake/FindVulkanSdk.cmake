# Locate the Vulkan SDK even when the VULKAN_SDK environment variable is unset
# (common on machines where the SDK installer ran without a shell restart).
#
# Priority:
#   1. DRESSI_VULKAN_SDK_DIR cache variable (manual override)
#   2. VULKAN_SDK environment variable
#   3. Newest version directory under C:/VulkanSDK (Windows convention)
#
# Result: ENV{VULKAN_SDK} is set so that CMake's FindVulkan module works,
# and the SDK's CMake package dir is appended to CMAKE_PREFIX_PATH so that
# find_package(glslang CONFIG) etc. resolve.

set(DRESSI_VULKAN_SDK_DIR "" CACHE PATH "Manual Vulkan SDK root override")

if(DRESSI_VULKAN_SDK_DIR)
    set(_dressi_sdk_root "${DRESSI_VULKAN_SDK_DIR}")
elseif(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    set(_dressi_sdk_root "$ENV{VULKAN_SDK}")
elseif(WIN32)
    file(GLOB _dressi_sdk_candidates "C:/VulkanSDK/*")
    set(_dressi_sdk_root "")
    foreach(_cand IN LISTS _dressi_sdk_candidates)
        if(IS_DIRECTORY "${_cand}" AND EXISTS "${_cand}/Include/vulkan/vulkan.h")
            if("${_cand}" STRGREATER "${_dressi_sdk_root}")
                set(_dressi_sdk_root "${_cand}")
            endif()
        endif()
    endforeach()
endif()

if(NOT _dressi_sdk_root OR NOT EXISTS "${_dressi_sdk_root}/Include/vulkan/vulkan.h")
    message(FATAL_ERROR
        "Vulkan SDK not found. Install it or set -DDRESSI_VULKAN_SDK_DIR=<path>.")
endif()

set(ENV{VULKAN_SDK} "${_dressi_sdk_root}")
list(APPEND CMAKE_PREFIX_PATH "${_dressi_sdk_root}/Lib/cmake")
set(DRESSI_VULKAN_SDK_ROOT "${_dressi_sdk_root}")
message(STATUS "Vulkan SDK: ${DRESSI_VULKAN_SDK_ROOT}")
