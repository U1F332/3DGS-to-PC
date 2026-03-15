#pragma once

#include "gs2pc/camera.h"
#include "gs2pc/types.h"

#include <vector>

namespace gs2pc {

struct RasterFrameInputs {
    const GaussianSet* gaussians = nullptr;
    const CameraState* camera = nullptr;
    const GaussConfig* gauss_config = nullptr;
    int sh_degree = 3;
    bool calculate_surface_distance = false;
};

struct RasterFrameOutputs {
    int rendered = 0;
    std::vector<float> color;
    std::vector<float> depth;
    std::vector<float> gauss_contributions;
    std::vector<float> gauss_surface_distances;
    std::vector<int> gauss_pixels;
};

[[nodiscard]] bool HasCudaRasterizer() noexcept;
[[nodiscard]] Status MarkVisible(const RasterFrameInputs& inputs, std::vector<bool>& present);
[[nodiscard]] Status RasterizeForward(const RasterFrameInputs& inputs, RasterFrameOutputs& outputs);

} // namespace gs2pc
