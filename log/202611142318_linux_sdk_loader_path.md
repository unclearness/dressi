# Find the Vulkan loader in the 1.4.350.1 Linux SDK layout

## Goal

The RX 9070 XT / UHD 730 host (PR #2 issuecomment-6784909878) could not
configure without `-DVulkan_LIBRARY=...`. In the 1.4.350.1 Linux tarball the
loader moved from `lib/` to `lib/VulkanLoader/lib/`, and with no
`libvulkan-dev` installed `find_package(Vulkan)` failed with "missing:
Vulkan_LIBRARY".

## What was done

- `cmake/FindVulkanSdk.cmake` appends `<sdk>/lib/VulkanLoader/lib` to
  `CMAKE_LIBRARY_PATH` when that directory exists. `CMAKE_LIBRARY_PATH` is
  searched before the system paths, so the SDK's loader keeps the priority
  `lib/` had in older tarballs.

## Verification

No Linux machine here, so the check reproduces FindVulkan's non-Windows call
(`find_library(Vulkan_LIBRARY NAMES vulkan HINTS $ENV{VULKAN_SDK}/lib)`) on
fake SDK trees with Linux library naming:

| layout | before | after |
| --- | --- | --- |
| 1.4.350.1 (`lib/VulkanLoader/lib/libvulkan.so`) | NOTFOUND | found |
| older (`lib/libvulkan.so`) | found | found, same file |

Windows: `cmake --preset msvc` and the Release build are unchanged.

## Open issues

- Not yet confirmed on a real 1.4.350.1 Linux install. The next Linux
  configure without the manual `-DVulkan_LIBRARY` is the test.
