#pragma once

#include "gs2pc/camera.h"
#include "gs2pc/config.h"
#include "gs2pc/types.h"

namespace gs2pc {

Status ConvertGaussiansToPointCloud(const GaussianSet& gaussians, const ConversionConfig& config,
                                    PointCloud& point_cloud, RenderStats* stats = nullptr);

} // namespace gs2pc
