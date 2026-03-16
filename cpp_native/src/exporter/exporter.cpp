#include "gs2pc/exporter.h"

#include <fstream>

namespace gs2pc {

Status ExportPointCloudPly(const PointCloud& point_cloud, const std::filesystem::path& output_path) {
    std::ofstream stream(output_path, std::ios::binary);
    if (!stream) {
        return Status::IoError("failed to open output file: " + output_path.string());
    }

    bool has_normals = false;
    for (const auto& point : point_cloud.points) {
        if (point.has_normal) {
            has_normals = true;
            break;
        }
    }

    stream << "ply\n";
    stream << "format ascii 1.0\n";
    stream << "element vertex " << point_cloud.points.size() << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    if (has_normals) {
        stream << "property float nx\n";
        stream << "property float ny\n";
        stream << "property float nz\n";
    }
    stream << "property uchar red\n";
    stream << "property uchar green\n";
    stream << "property uchar blue\n";
    stream << "end_header\n";

    for (const auto& point : point_cloud.points) {
        const auto to_channel = [](float value) -> int {
            if (value <= 0.0f) {
                return 0;
            }
            if (value >= 1.0f) {
                return 255;
            }
            return static_cast<int>(value * 255.0f);
        };

        stream << point.position.x << ' ' << point.position.y << ' ' << point.position.z << ' ';

        if (has_normals) {
            const Vec3f normal = point.has_normal ? point.normal : Vec3f{};
            stream << normal.x << ' ' << normal.y << ' ' << normal.z << ' ';
        }

        stream << to_channel(point.color[0]) << ' ' << to_channel(point.color[1]) << ' ' << to_channel(point.color[2])
               << '\n';
    }

    if (!stream) {
        return Status::IoError("failed while writing output file: " + output_path.string());
    }

    return Status::Ok();
}

} // namespace gs2pc
