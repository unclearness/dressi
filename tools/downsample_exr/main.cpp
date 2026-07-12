// Bakes a small equirectangular env map for the Android app: box-downsample
// an EXR to <= max_width and save half-float/ZIP. The PBS examples
// downsample their env input to <= 512 px anyway, so a pre-baked 512-wide
// map is an equivalent input at ~1% of the 4k file size.
//
//   downsample_exr <in.exr> <out.exr> [max_width=512]

#include <cstdio>
#include <cstdlib>
#include <string>

#include "../../examples/common/asset_utils.h"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: downsample_exr <in.exr> <out.exr> "
                     "[max_width=512]\n");
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const uint32_t max_w = argc > 3 ? uint32_t(std::atoi(argv[3])) : 512u;

    dressi::CpuImage img = dressi_examples::LoadImageExr(in_path);
    std::printf("%s: %ux%u\n", in_path.c_str(), img.width, img.height);
    while (img.width > max_w) {
        img = dressi_examples::DownsampleHalf(img);
    }
    dressi_examples::SaveImageExr(out_path, img);
    std::printf("%s: %ux%u (half float, zip)\n", out_path.c_str(), img.width,
                img.height);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
