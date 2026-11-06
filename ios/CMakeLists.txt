# iOS apps: DressiExamples (all examples, the counterpart of android/) and
# DressiTests (every gtest suite, since an iOS binary only runs as an app).
# Signed by Xcode: pass -DDRESSI_IOS_TEAM=<team id> for device builds.
set(DRESSI_IOS_TEAM "" CACHE STRING "Apple development team id")
set(DRESSI_IOS_BUNDLE_PREFIX "org.dressi" CACHE STRING "Bundle id prefix")
set(DRESSI_IOS_DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH
    "Desktop-provisioned data/ to bundle into DressiExamples")

function(dressi_ios_app target bundle_suffix)
    set_target_properties(${target} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
        MACOSX_BUNDLE_BUNDLE_NAME "${target}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "${DRESSI_IOS_BUNDLE_PREFIX}.${bundle_suffix}"
        MACOSX_BUNDLE_BUNDLE_VERSION "1"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
            "${DRESSI_IOS_BUNDLE_PREFIX}.${bundle_suffix}"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${DRESSI_IOS_TEAM}")
    target_compile_options(${target} PRIVATE -fobjc-arc)
endfunction()

add_executable(DressiExamples MACOSX_BUNDLE app_main.mm)
target_include_directories(DressiExamples PRIVATE
    "${CMAKE_SOURCE_DIR}/examples/common")
target_link_libraries(DressiExamples PRIVATE dressi_examples_core)
dressi_ios_app(DressiExamples examples)

# Same curated subset as the Android app; the 97 MB 4k EXR stays out.
foreach(_asset bunny Avocado DamagedHelmet)
    if(EXISTS "${DRESSI_IOS_DATA_DIR}/${_asset}")
        add_custom_command(TARGET DressiExamples POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${DRESSI_IOS_DATA_DIR}/${_asset}"
                "$<TARGET_BUNDLE_CONTENT_DIR:DressiExamples>/data/${_asset}")
    else()
        message(WARNING "dressi: ${_asset} missing from ${DRESSI_IOS_DATA_DIR}; "
                        "examples using it will fail in the app")
    endif()
endforeach()
if(EXISTS "${DRESSI_IOS_DATA_DIR}/suburban_garden_512.exr")
    add_custom_command(TARGET DressiExamples POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${DRESSI_IOS_DATA_DIR}/suburban_garden_512.exr"
            "$<TARGET_BUNDLE_CONTENT_DIR:DressiExamples>/data/suburban_garden_512.exr")
else()
    message(WARNING "dressi: suburban_garden_512.exr missing; see README-ios.md")
endif()

if(DRESSI_BUILD_TESTS)
    add_executable(DressiTests MACOSX_BUNDLE test_host.mm
        ${DRESSI_TESTS_CPU_SOURCES}
        ${DRESSI_TESTS_GPU_SOURCES}
        ${DRESSI_TESTS_SMOKE_SOURCES})
    target_include_directories(DressiTests PRIVATE
        "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tests")
    target_link_libraries(DressiTests PRIVATE dressi GTest::gtest)
    dressi_ios_app(DressiTests tests)
endif()
