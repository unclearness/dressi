#include "desktop_host.h"

#include "vk_viewer.h"

namespace dressi_examples {

namespace {

class DesktopViewer : public IViewer {
public:
    DesktopViewer(uint32_t width, uint32_t height, const std::string& title)
        : m_viewer(width, height, title) {}

    bool valid() const override { return m_viewer.valid(); }
    bool update(const dressi::CpuImage& img) override {
        return m_viewer.update(img);
    }
    void setTitle(const std::string& title) override {
        m_viewer.setTitle(title);
    }
    void setPosition(int x, int y) override { m_viewer.setPosition(x, y); }

private:
    VkViewer m_viewer;
};

}  // namespace

std::unique_ptr<IViewer> DesktopHost::makeViewer(uint32_t width,
                                                 uint32_t height,
                                                 const std::string& title) {
    return std::make_unique<DesktopViewer>(width, height, title);
}

}  // namespace dressi_examples
