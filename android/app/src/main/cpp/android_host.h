#ifndef DRESSI_ANDROID_HOST_H
#define DRESSI_ANDROID_HOST_H

#include <android/native_window.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "example_host.h"

namespace dressi_android {

// Callbacks implemented in jni_bridge.cpp (dispatch into the Kotlin
// listener; safe to call from any attached thread).
void NotifyStreamsChanged(const std::vector<std::string>& titles);

// The app's single Surface plus the set of viewer "streams" the running
// example registered (desktop shows N windows; Android shows one selected
// stream on the SurfaceView). All access is serialized by one mutex, so
// surface teardown (surfaceDestroyed) blocks until an in-flight blit is
// done, as the Android Surface contract requires.
class SurfaceState {
public:
    static SurfaceState& instance();

    // Takes ownership of an already-acquired window (or nullptr to detach).
    void setSurface(ANativeWindow* window);
    void selectStream(int index);

    int registerStream(const std::string& title);  // -> stream id
    void clearStreams();

    // Caches this stream's latest frame (so a stream can be shown even after
    // the example loop has stopped calling update), then converts the float
    // [0,1] CpuImage (1 or 3 channels) to RGBA8, nearest-scales with
    // letterboxing, and posts it when the stream is selected. Always returns
    // true (a non-selected stream keeps the example running).
    bool blit(int stream_id, const dressi::CpuImage& img);

private:
    SurfaceState() = default;
    // Posts an already-RGBA-convertible float image to the current window.
    // Caller must hold m_mutex and guarantee m_window != nullptr.
    void drawLocked(const dressi::CpuImage& img);

    std::mutex m_mutex;
    ANativeWindow* m_window = nullptr;
    int m_selected = 0;
    std::vector<std::string> m_titles;
    // Latest frame per stream id (parallel to m_titles); empty until a stream
    // first blits. Lets selectStream / surface re-attach redraw a finished run.
    std::vector<dressi::CpuImage> m_frames;
};

class AndroidHost : public dressi_examples::ExampleHost {
public:
    explicit AndroidHost(const std::atomic<bool>& cancel) : m_cancel(cancel) {}

    std::unique_ptr<dressi_examples::IViewer> makeViewer(
            uint32_t width, uint32_t height,
            const std::string& title) override;

    bool cancelled() const override {
        return m_cancel.load(std::memory_order_relaxed);
    }

private:
    const std::atomic<bool>& m_cancel;
};

// Installs the spdlog default logger: logcat (tag "dressi") + a callback
// sink forwarding formatted lines to the Kotlin listener.
void InstallLogSinks();

}  // namespace dressi_android

#endif  // DRESSI_ANDROID_HOST_H
