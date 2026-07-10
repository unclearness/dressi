#ifndef DRESSI_EXAMPLES_SOBOL_H
#define DRESSI_EXAMPLES_SOBOL_H

#include <array>
#include <cstdint>

namespace dressi_examples {

// i-th point of the standard 2D Sobol (0,2)-sequence in [0,1)^2:
// dimension 0 is the base-2 radical inverse (bit reversal), dimension 1
// uses the classic direction-vector recurrence v_j = v_{j-1} ^ (v_{j-1}>>1).
inline std::array<float, 2> Sobol2D(uint32_t index) {
    uint32_t rev = index;
    rev = (rev << 16) | (rev >> 16);
    rev = ((rev & 0x00ff00ffu) << 8) | ((rev & 0xff00ff00u) >> 8);
    rev = ((rev & 0x0f0f0f0fu) << 4) | ((rev & 0xf0f0f0f0u) >> 4);
    rev = ((rev & 0x33333333u) << 2) | ((rev & 0xccccccccu) >> 2);
    rev = ((rev & 0x55555555u) << 1) | ((rev & 0xaaaaaaaau) >> 1);

    uint32_t dim1 = 0;
    uint32_t v = 1u << 31;
    for (uint32_t bits = index; bits != 0; bits >>= 1) {
        if (bits & 1u) {
            dim1 ^= v;
        }
        v ^= v >> 1;
    }

    constexpr float kInv = 1.f / 4294967296.f;  // 2^-32
    return {float(rev) * kInv, float(dim1) * kInv};
}

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_SOBOL_H
