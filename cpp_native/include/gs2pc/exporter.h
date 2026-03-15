#pragma once

#include "gs2pc/types.h"

#include <filesystem>

namespace gs2pc {

Status ExportPointCloudPly(const PointCloud& point_cloud, const std::filesystem::path& output_path);

} // namespace gs2pc
