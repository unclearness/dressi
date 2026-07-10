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
