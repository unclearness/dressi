#include "android_host.h"

#include <spdlog/sinks/android_sink.h>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dressi_android {

void NotifyLog(const std::string& line);  // jni_bridge.cpp

SurfaceState& SurfaceState::instance() {
    static SurfaceState state;
    return state;
}

void SurfaceState::setSurface(ANativeWindow* window) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_window) {
        ANativeWindow_release(m_window);
    }
    m_window = window;
    if (m_window) {
        // Fixed pixel layout for the CPU blit; the compositor scales the
        // buffer to the view, we letterbox inside it ourselves.
        ANativeWindow_setBuffersGeometry(m_window, 0, 0,
                                         WINDOW_FORMAT_RGBA_8888);
        // A run's SurfaceView is destroyed/recreated when Back/Start toggle
        // its visibility, but the swapchain buffers keep the previous
        // example's last frame until the new run blits (seconds away during
        // the first build). Clear every buffer to black so no stale screen
        // shows through. Post a few times to cover double/triple buffering.
        for (int i = 0; i < 3; i++) {
            ANativeWindow_Buffer buf;
            if (ANativeWindow_lock(m_window, &buf, nullptr) != 0) {
                break;
            }
            auto* pixels = static_cast<uint32_t*>(buf.bits);
            for (uint32_t y = 0; y < uint32_t(buf.height); y++) {
                uint32_t* row = pixels + size_t(y) * uint32_t(buf.stride);
                std::fill(row, row + uint32_t(buf.width), 0xFF000000u);
            }
            ANativeWindow_unlockAndPost(m_window);
        }
        // Restore the selected stream's last frame onto the fresh surface
        // (e.g. after the SurfaceView is recreated post-run), so switching
        // back does not leave the view black until the next blit.
        if (m_selected == kAllStream) {
            drawAllLocked();
        } else if (m_selected >= 0 && size_t(m_selected) < m_frames.size() &&
                   m_frames[size_t(m_selected)].width != 0) {
            drawLocked(m_frames[size_t(m_selected)]);
        }
    }
}

void SurfaceState::selectStream(int index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    // The UI's stream list ends with the synthetic "all" entry (see
    // registerStream); its index is the number of real streams.
    if (index >= 0 && size_t(index) == m_titles.size() &&
        m_titles.size() >= 2) {
        m_selected = kAllStream;
        if (m_window) {
            drawAllLocked();
        }
        return;
    }
    m_selected = index;
    // Immediately show the newly selected stream's cached frame. Without this
    // the surface stays on the previous stream until the next update() -- and
    // once the optimization has finished there are no more update() calls, so
    // the stream buttons would appear dead.
    if (m_window && index >= 0 && size_t(index) < m_frames.size() &&
        m_frames[size_t(index)].width != 0) {
        drawLocked(m_frames[size_t(index)]);
    }
}

int SurfaceState::registerStream(const std::string& title) {
    std::vector<std::string> titles;
    int id = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        id = int(m_titles.size());
        m_titles.push_back(title);
        m_frames.emplace_back();  // empty until this stream first blits
        // Default to the synthetic "all" (tiled) view as soon as a second
        // stream exists; the UI mirrors this (it selects the last entry of
        // the notified titles) and can override it via selectStream.
        if (id == 1) {
            m_selected = kAllStream;
        }
        titles = m_titles;
    }
    // With >= 2 streams the UI also gets a synthetic "all" entry that tiles
    // every stream on the one surface (selectStream maps its index, which
    // equals the real-stream count, to kAllStream).
    if (titles.size() >= 2) {
        titles.push_back("all");
    }
    NotifyStreamsChanged(titles);
    return id;
}

void SurfaceState::clearStreams() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_titles.clear();
        m_frames.clear();
        m_selected = 0;
    }
    NotifyStreamsChanged({});
}

bool SurfaceState::blit(int stream_id, const dressi::CpuImage& img) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (img.width == 0 || img.height == 0) {
        return true;
    }
    // Cache every stream's latest frame so selectStream (and a post-run
    // surface re-attach) can redraw it after the example loop has stopped.
    if (stream_id >= 0 && size_t(stream_id) < m_frames.size()) {
        m_frames[size_t(stream_id)] = img;
    }
    if (!m_window) {
        return true;
    }
    if (m_selected == kAllStream) {
        drawAllLocked();  // any stream's new frame refreshes the composite
        return true;
    }
    if (stream_id != m_selected) {
        return true;  // stream not visible; keep the example running
    }
    drawLocked(img);
    return true;
}

void SurfaceState::drawAllLocked() {
    // Collect the streams that have blitted at least once, in registration
    // order (= the order of the UI buttons).
    std::vector<const dressi::CpuImage*> frames;
    for (const dressi::CpuImage& f : m_frames) {
        if (f.width != 0 && f.height != 0) {
            frames.push_back(&f);
        }
    }
    if (frames.empty()) {
        return;
    }
    if (frames.size() == 1) {
        drawLocked(*frames.front());
        return;
    }
    // Vertical stack (portrait-friendly): every frame nearest-scaled to the
    // widest stream's width, aspect preserved, with a thin black separator.
    constexpr uint32_t kGap = 2;
    uint32_t canvas_w = 0;
    for (const dressi::CpuImage* f : frames) {
        canvas_w = std::max(canvas_w, f->width);
    }
    std::vector<uint32_t> row_h;
    uint32_t canvas_h = kGap * uint32_t(frames.size() - 1);
    for (const dressi::CpuImage* f : frames) {
        row_h.push_back(std::max(
                1u, uint32_t(uint64_t(f->height) * canvas_w / f->width)));
        canvas_h += row_h.back();
    }
    if (m_all_canvas.width != canvas_w || m_all_canvas.height != canvas_h ||
        m_all_canvas.channels != 3) {
        m_all_canvas = dressi::CpuImage(canvas_w, canvas_h, 3, 0.f);
    }
    uint32_t oy = 0;
    for (size_t i = 0; i < frames.size(); i++) {
        const dressi::CpuImage& f = *frames[i];
        const bool rgb = f.channels >= 3;
        for (uint32_t y = 0; y < row_h[i]; y++) {
            const uint32_t sy = std::min(
                    f.height - 1, uint32_t(uint64_t(y) * f.height / row_h[i]));
            for (uint32_t x = 0; x < canvas_w; x++) {
                const uint32_t sx = std::min(
                        f.width - 1,
                        uint32_t(uint64_t(x) * f.width / canvas_w));
                const size_t p =
                        (size_t(sy) * f.width + sx) * f.channels;
                m_all_canvas.at(x, oy + y, 0) = f.data[p];
                m_all_canvas.at(x, oy + y, 1) = rgb ? f.data[p + 1]
                                                    : f.data[p];
                m_all_canvas.at(x, oy + y, 2) = rgb ? f.data[p + 2]
                                                    : f.data[p];
            }
        }
        oy += row_h[i];
        if (i + 1 < frames.size()) {
            for (uint32_t y = 0; y < kGap; y++) {
                for (uint32_t x = 0; x < canvas_w; x++) {
                    m_all_canvas.at(x, oy + y, 0) = 0.f;
                    m_all_canvas.at(x, oy + y, 1) = 0.f;
                    m_all_canvas.at(x, oy + y, 2) = 0.f;
                }
            }
            oy += kGap;
        }
    }
    drawLocked(m_all_canvas);
}

void SurfaceState::drawLocked(const dressi::CpuImage& img) {
    ANativeWindow_Buffer buf;
    if (ANativeWindow_lock(m_window, &buf, nullptr) != 0) {
        return;
    }
    auto* pixels = static_cast<uint32_t*>(buf.bits);
    const uint32_t bw = uint32_t(buf.width), bh = uint32_t(buf.height);
    // Letterbox: integer-friendly nearest scale preserving aspect
    const float scale = std::min(float(bw) / float(img.width),
                                 float(bh) / float(img.height));
    const uint32_t dw = std::max(1u, uint32_t(float(img.width) * scale));
    const uint32_t dh = std::max(1u, uint32_t(float(img.height) * scale));
    const uint32_t ox = (bw - dw) / 2, oy = (bh - dh) / 2;
    const bool rgb = img.channels >= 3;
    for (uint32_t y = 0; y < bh; y++) {
        uint32_t* row = pixels + size_t(y) * uint32_t(buf.stride);
        if (y < oy || y >= oy + dh) {
            std::fill(row, row + bw, 0xFF000000u);
            continue;
        }
        const uint32_t sy =
                std::min(img.height - 1,
                         uint32_t(float(y - oy) / float(dh) *
                                  float(img.height)));
        for (uint32_t x = 0; x < bw; x++) {
            if (x < ox || x >= ox + dw) {
                row[x] = 0xFF000000u;
                continue;
            }
            const uint32_t sx =
                    std::min(img.width - 1,
                             uint32_t(float(x - ox) / float(dw) *
                                      float(img.width)));
            const size_t p = (size_t(sy) * img.width + sx) * img.channels;
            const auto to8 = [&](float v) {
                return uint32_t(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
            };
            const uint32_t r = to8(img.data[p]);
            const uint32_t g = rgb ? to8(img.data[p + 1]) : r;
            const uint32_t b = rgb ? to8(img.data[p + 2]) : r;
            row[x] = 0xFF000000u | (b << 16) | (g << 8) | r;
        }
    }
    ANativeWindow_unlockAndPost(m_window);
}

namespace {

class AndroidViewer : public dressi_examples::IViewer {
public:
    explicit AndroidViewer(const std::string& title)
        : m_stream_id(SurfaceState::instance().registerStream(title)) {}

    bool valid() const override { return true; }
    bool update(const dressi::CpuImage& img) override {
        return SurfaceState::instance().blit(m_stream_id, img);
    }
    void setTitle(const std::string& title) override {
        (void)title;  // stream titles stay stable; iter counters go to logs
    }
    void setPosition(int, int) override {}

private:
    int m_stream_id;
};

}  // namespace

std::unique_ptr<dressi_examples::IViewer> AndroidHost::makeViewer(
        uint32_t width, uint32_t height, const std::string& title) {
    (void)width;
    (void)height;  // the blit letterboxes into whatever surface exists
    return std::make_unique<AndroidViewer>(title);
}

void InstallLogSinks() {
    auto logcat = std::make_shared<spdlog::sinks::android_sink_mt>("dressi");
    auto ui = std::make_shared<spdlog::sinks::callback_sink_mt>(
            [](const spdlog::details::log_msg& msg) {
                NotifyLog(std::string(msg.payload.begin(),
                                      msg.payload.end()));
            });
    auto logger = std::make_shared<spdlog::logger>(
            "dressi", spdlog::sinks_init_list{logcat, ui});
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
}

}  // namespace dressi_android
