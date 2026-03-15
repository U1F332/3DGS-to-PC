#pragma once

#include "gs2pc/config.h"

#include <string>
#include <vector>

namespace gs2pc {

struct CameraState {
    CameraIntrinsics intrinsics;
    CameraPose pose;
    RenderConfig render;
    RenderMask mask;
};

struct CameraFrame {
    std::string name;
    CameraState state;
};

inline Status ValidateCameraState(const CameraState& camera) {
    if (camera.render.image_width <= 0 || camera.render.image_height <= 0) {
        return Status::InvalidArgument("camera image dimensions must be greater than zero");
    }

    if (camera.intrinsics.tan_fov_x <= 0.0f || camera.intrinsics.tan_fov_y <= 0.0f) {
        return Status::InvalidArgument("camera tan_fov values must be greater than zero");
    }

    return Status::Ok();
}

Status LoadCameraFrames(const std::filesystem::path& transform_path, const ConversionConfig& config,
                        std::vector<CameraFrame>& frames);

} // namespace gs2pc
