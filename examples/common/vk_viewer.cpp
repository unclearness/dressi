#include "vk_viewer.h"

#include <glad/vulkan.h>
// GLFW after glad: VK_VERSION_1_0 is visible so glfw3.h declares its Vulkan
// helpers without pulling in another vulkan.h
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace dressi_examples {

using dressi::CpuImage;

namespace {

GLADapiproc GlfwVkLoader(void* instance, const char* name) {
    return reinterpret_cast<GLADapiproc>(glfwGetInstanceProcAddress(
            reinterpret_cast<VkInstance>(instance), name));
}

}  // namespace

struct VkViewer::Impl {
    uint32_t width = 0;
    uint32_t height = 0;
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat sc_format = VK_FORMAT_B8G8R8A8_UNORM;
    std::vector<VkImage> sc_images;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void* staging_map = nullptr;
    VkSemaphore sem_acquire = VK_NULL_HANDLE;
    VkSemaphore sem_render = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;

    bool init(const std::string& title);
    void destroy();
};

bool VkViewer::Impl::init(const std::string& title) {
    if (!glfwInit() || !glfwVulkanSupported()) {
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(int(width), int(height), title.c_str(),
                              nullptr, nullptr);
    if (!window) {
        return false;
    }

    gladLoadVulkanUserPtr(nullptr, GlfwVkLoader, nullptr);

    uint32_t n_glfw_exts = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&n_glfw_exts);
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dressi_viewer";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = n_glfw_exts;
    ici.ppEnabledExtensionNames = glfw_exts;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        return false;
    }
    gladLoadVulkanUserPtr(nullptr, GlfwVkLoader, instance);

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
        return false;
    }

    uint32_t n_phys = 0;
    vkEnumeratePhysicalDevices(instance, &n_phys, nullptr);
    std::vector<VkPhysicalDevice> devices(n_phys);
    vkEnumeratePhysicalDevices(instance, &n_phys, devices.data());
    for (auto d : devices) {
        uint32_t n_qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &n_qf, nullptr);
        std::vector<VkQueueFamilyProperties> qf(n_qf);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &n_qf, qf.data());
        for (uint32_t i = 0; i < n_qf; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                phys = d;
                queue_family = i;
                break;
            }
        }
        if (phys) {
            break;
        }
    }
    if (!phys) {
        return false;
    }
    gladLoadVulkanUserPtr(phys, GlfwVkLoader, instance);

    const float prio = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        return false;
    }
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);
    uint32_t n_fmts = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n_fmts, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n_fmts);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n_fmts, fmts.data());
    VkSurfaceFormatKHR fmt = fmts[0];
    for (const auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM ||
            f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            fmt = f;
            break;
        }
    }
    sc_format = fmt.format;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = surface;
    sci.minImageCount = std::max(2u, caps.minImageCount);
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = {width, height};
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain) !=
        VK_SUCCESS) {
        return false;
    }
    uint32_t n_imgs = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &n_imgs, nullptr);
    sc_images.resize(n_imgs);
    vkGetSwapchainImagesKHR(device, swapchain, &n_imgs, sc_images.data());

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queue_family;
    vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cbai, &cmd);

    const VkDeviceSize staging_size = VkDeviceSize(width) * height * 4;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = staging_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bci, nullptr, &staging);
    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(device, staging, &mem_req);
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    uint32_t mem_type = UINT32_MAX;
    const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & want) == want) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX) {
        return false;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mem_req.size;
    mai.memoryTypeIndex = mem_type;
    vkAllocateMemory(device, &mai, nullptr, &staging_mem);
    vkBindBufferMemory(device, staging, staging_mem, 0);
    vkMapMemory(device, staging_mem, 0, staging_size, 0, &staging_map);

    VkSemaphoreCreateInfo semci{};
    semci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(device, &semci, nullptr, &sem_acquire);
    vkCreateSemaphore(device, &semci, nullptr, &sem_render);
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fci, nullptr, &fence);

    ok = true;
    return true;
}

void VkViewer::Impl::destroy() {
    if (device) {
        vkDeviceWaitIdle(device);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (sem_acquire) vkDestroySemaphore(device, sem_acquire, nullptr);
        if (sem_render) vkDestroySemaphore(device, sem_render, nullptr);
        if (staging_map) vkUnmapMemory(device, staging_mem);
        if (staging) vkDestroyBuffer(device, staging, nullptr);
        if (staging_mem) vkFreeMemory(device, staging_mem, nullptr);
        if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) {
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
    if (window) {
        glfwDestroyWindow(window);
    }
}

VkViewer::VkViewer(uint32_t width, uint32_t height, const std::string& title)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->width = width;
    m_impl->height = height;
    if (!m_impl->init(title)) {
        m_impl->destroy();
        m_impl->ok = false;
    }
}

VkViewer::~VkViewer() {
    if (m_impl->ok) {
        m_impl->destroy();
    }
}

bool VkViewer::valid() const {
    return m_impl->ok;
}

void VkViewer::setTitle(const std::string& title) {
    if (m_impl->ok) {
        glfwSetWindowTitle(m_impl->window, title.c_str());
    }
}

void VkViewer::setPosition(int x, int y) {
    if (m_impl->ok) {
        glfwSetWindowPos(m_impl->window, x, y);
    }
}

bool VkViewer::update(const CpuImage& img) {
    Impl& im = *m_impl;
    if (!im.ok) {
        return false;
    }
    glfwPollEvents();
    if (glfwWindowShouldClose(im.window)) {
        return false;
    }

    // Convert to the swapchain byte order, nearest-scaled to the window
    const bool bgr = im.sc_format == VK_FORMAT_B8G8R8A8_UNORM;
    auto* dst = static_cast<unsigned char*>(im.staging_map);
    for (uint32_t y = 0; y < im.height; y++) {
        const uint32_t sy =
                std::min(uint32_t(uint64_t(y) * img.height / im.height),
                         img.height - 1);
        for (uint32_t x = 0; x < im.width; x++) {
            const uint32_t sx =
                    std::min(uint32_t(uint64_t(x) * img.width / im.width),
                             img.width - 1);
            unsigned char rgb[3];
            for (uint32_t c = 0; c < 3; c++) {
                const float v = img.at(sx, sy, img.channels == 1 ? 0 : c);
                rgb[c] = (unsigned char)(std::clamp(v, 0.f, 1.f) * 255.f +
                                         0.5f);
            }
            unsigned char* px = dst + (size_t(y) * im.width + x) * 4;
            px[0] = bgr ? rgb[2] : rgb[0];
            px[1] = rgb[1];
            px[2] = bgr ? rgb[0] : rgb[2];
            px[3] = 255;
        }
    }

    uint32_t img_idx = 0;
    VkResult res = vkAcquireNextImageKHR(im.device, im.swapchain, UINT64_MAX,
                                         im.sem_acquire, VK_NULL_HANDLE,
                                         &img_idx);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        return true;  // skip the frame (e.g. minimized)
    }
    VkImage target = im.sc_images[img_idx];

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(im.cmd, 0);
    vkBeginCommandBuffer(im.cmd, &begin);

    VkImageMemoryBarrier to_dst{};
    to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_dst.srcAccessMask = 0;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = target;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(im.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_dst);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {im.width, im.height, 1};
    vkCmdCopyBufferToImage(im.cmd, im.staging, target,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier to_present = to_dst;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(im.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &to_present);
    vkEndCommandBuffer(im.cmd);

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &im.sem_acquire;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &im.cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &im.sem_render;
    vkQueueSubmit(im.queue, 1, &submit, im.fence);

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &im.sem_render;
    present.swapchainCount = 1;
    present.pSwapchains = &im.swapchain;
    present.pImageIndices = &img_idx;
    vkQueuePresentKHR(im.queue, &present);

    vkWaitForFences(im.device, 1, &im.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(im.device, 1, &im.fence);
    return true;
}

}  // namespace dressi_examples
