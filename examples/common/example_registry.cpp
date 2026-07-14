#include "example_host.h"

#include "../image_fitting/run.h"
#include "../pbr_envmap_optimization/run.h"
#include "../pbr_material_optimization/run.h"
#include "../pbr_shading/run.h"
#include "../shape_texture_optimization/run.h"
#include "../silhouette_optimization/run.h"
#include "../texture_optimization/run.h"

namespace dressi_examples {

const std::vector<ExampleInfo>& GetExampleRegistry() {
    static const std::vector<ExampleInfo> registry = {
            {"image_fitting", &RunImageFitting, false},
            {"texture_optimization", &RunTextureOptimization, false},
            {"silhouette_optimization", &RunSilhouetteOptimization, true},
            {"shape_texture_optimization", &RunShapeTextureOptimization, true},
            {"pbr_shading", &RunPbrShading, false},
            {"pbr_material_optimization", &RunPbrMaterialOptimization, false},
            {"pbr_envmap_optimization", &RunPbrEnvmapOptimization, false},
    };
    return registry;
}

}  // namespace dressi_examples
