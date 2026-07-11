#ifndef DRESSI_TYPES_H
#define DRESSI_TYPES_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dressi {

// Variable data types (2D images of these element types)
enum VType {
    FLOAT,
    VEC2,
    VEC3,
    VEC4,
    MAT2,
    MAT3,
    MAT4,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    UINT,
};

// Shader type of a Function
enum ShaderType {
    FRAG,
    COMP,
    RASTER,
};

struct ImgSize {
    uint32_t w = 1;
    uint32_t h = 1;
    bool operator==(const ImgSize&) const = default;
    bool isUniform() const { return w == 1 && h == 1; }
};

// 2-D tiling of oversized 1-D data images. A {N,1} texelFetch image whose
// width N exceeds the device's maxImageDimension2D cannot be created (NVIDIA's
// limit is 32768, Intel iGPUs report 16384). Such an image is instead stored
// as {W, ceil(N/W)} with W a power of two <= the limit, so neither extent
// exceeds it. Addressing stays linear -- element i lives at
// (i & (W-1), i >> log2(W)), matching the row-major image memory -- so a
// shader is layout-agnostic: when N fits, W == N and it degrades exactly to
// the untiled ivec2(i, 0) fetch. `max_dim == 0` (the default sentinel used off
// the device path) disables tiling entirely.
inline uint32_t TileWidth(uint32_t max_dim) {
    uint32_t w = 1;
    while ((w << 1) <= max_dim) {
        w <<= 1;
    }
    return w;  // largest power of two <= max_dim
}

inline bool NeedsTiling(ImgSize s, uint32_t max_dim) {
    return max_dim != 0 && s.h == 1 && s.w > max_dim;
}

inline ImgSize PhysImgSize(ImgSize logical, uint32_t max_dim) {
    if (!NeedsTiling(logical, max_dim)) {
        return logical;
    }
    const uint32_t w = TileWidth(max_dim);
    return {w, (logical.w + w - 1) / w};
}

// Simple CPU-side buffer of a 2D image (row-major, tightly packed logical
// channels; e.g. VEC3 -> channels == 3)
struct CpuImage {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    std::vector<float> data;

    CpuImage() = default;
    CpuImage(uint32_t w, uint32_t h, uint32_t c, float fill = 0.f)
        : width(w), height(h), channels(c), data(size_t(w) * h * c, fill) {}
    ImgSize getImgSize() const { return {width, height}; }
    float& at(uint32_t x, uint32_t y, uint32_t c) {
        return data[(size_t(y) * width + x) * channels + c];
    }
    float at(uint32_t x, uint32_t y, uint32_t c) const {
        return data[(size_t(y) * width + x) * channels + c];
    }
};

// Non-owning view of CPU image data with CpuImage's layout (row-major,
// tightly packed logical channels). Lets language bindings upload borrowed
// buffers without an intermediate CpuImage copy.
struct CpuImageView {
    const float* data = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;

    CpuImageView() = default;
    CpuImageView(const float* d, uint32_t w, uint32_t h, uint32_t c)
        : data(d), width(w), height(h), channels(c) {}
    explicit CpuImageView(const CpuImage& img)
        : data(img.data.data()),
          width(img.width),
          height(img.height),
          channels(img.channels) {}
    size_t numElems() const { return size_t(width) * height * channels; }
};

class DressiError : public std::runtime_error {
public:
    explicit DressiError(const std::string& msg) : std::runtime_error(msg) {}
};

[[noreturn]] void ThrowDressiError(const std::string& msg, const char* file,
                                   int line);

#define DRESSI_CHECK(cond, msg)                                   \
    do {                                                          \
        if (!(cond)) {                                            \
            ::dressi::ThrowDressiError((msg), __FILE__, __LINE__); \
        }                                                         \
    } while (false)

// Helpers over VType
uint32_t NumComponents(VType vtype);
bool IsIntVType(VType vtype);
bool IsMatrixVType(VType vtype);
bool IsVectorVType(VType vtype);  // VEC2..VEC4 / IVEC2..IVEC4
const char* GlslTypeName(VType vtype);
VType VecVTypeOf(uint32_t n_components, bool is_int = false);

}  // namespace dressi

#endif  // DRESSI_TYPES_H
