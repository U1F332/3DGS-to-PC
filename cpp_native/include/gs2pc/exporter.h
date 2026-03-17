#pragma once

#include "gs2pc/config.h"
#include "gs2pc/types.h"

#include <filesystem>

namespace gs2pc {

struct ExportTimings {
    long long header_write_ms = 0;
    long long vertex_write_ms = 0;
    long long flush_ms = 0;
    long long total_ms = 0;
};

Status ExportPointCloudPly(const PointCloud& point_cloud, const std::filesystem::path& output_path,
                           PlyExportFormat format = PlyExportFormat::Binary,
                           ExportTimings* timings = nullptr);

} // namespace gs2pc
