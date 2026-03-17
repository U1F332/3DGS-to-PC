#include "gs2pc/exporter.h"

#include <chrono>
#include <cstdint>
#include <fstream>

namespace gs2pc {
namespace {

inline std::uint8_t ToChannel(float value) {
    if (value <= 0.0f) {
        return static_cast<std::uint8_t>(0);
    }
    if (value >= 1.0f) {
        return static_cast<std::uint8_t>(255);
    }
    return static_cast<std::uint8_t>(value * 255.0f);
}

} // namespace

Status ExportPointCloudPly(const PointCloud& point_cloud, const std::filesystem::path& output_path,
                           PlyExportFormat format,
                           ExportTimings* timings) {
    const auto total_start = std::chrono::steady_clock::now();

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

    const auto header_start = std::chrono::steady_clock::now();
    stream << "ply\n";
    if (format == PlyExportFormat::Binary) {
        stream << "format binary_little_endian 1.0\n";
    } else {
        stream << "format ascii 1.0\n";
    }
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
    const auto header_end = std::chrono::steady_clock::now();

    const auto vertex_start = std::chrono::steady_clock::now();
    if (format == PlyExportFormat::Binary) {
        for (const auto& point : point_cloud.points) {
            const float x = point.position.x;
            const float y = point.position.y;
            const float z = point.position.z;
            stream.write(reinterpret_cast<const char*>(&x), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&y), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&z), sizeof(float));

            if (has_normals) {
                const Vec3f normal = point.has_normal ? point.normal : Vec3f{};
                const float nx = normal.x;
                const float ny = normal.y;
                const float nz = normal.z;
                stream.write(reinterpret_cast<const char*>(&nx), sizeof(float));
                stream.write(reinterpret_cast<const char*>(&ny), sizeof(float));
                stream.write(reinterpret_cast<const char*>(&nz), sizeof(float));
            }

            const std::uint8_t r = ToChannel(point.color[0]);
            const std::uint8_t g = ToChannel(point.color[1]);
            const std::uint8_t b = ToChannel(point.color[2]);
            stream.write(reinterpret_cast<const char*>(&r), sizeof(std::uint8_t));
            stream.write(reinterpret_cast<const char*>(&g), sizeof(std::uint8_t));
            stream.write(reinterpret_cast<const char*>(&b), sizeof(std::uint8_t));
        }
    } else {
        for (const auto& point : point_cloud.points) {
            stream << point.position.x << ' ' << point.position.y << ' ' << point.position.z << ' ';

            if (has_normals) {
                const Vec3f normal = point.has_normal ? point.normal : Vec3f{};
                stream << normal.x << ' ' << normal.y << ' ' << normal.z << ' ';
            }

            stream << static_cast<int>(ToChannel(point.color[0])) << ' ' << static_cast<int>(ToChannel(point.color[1]))
                   << ' ' << static_cast<int>(ToChannel(point.color[2])) << '\n';
        }
    }
    const auto vertex_end = std::chrono::steady_clock::now();

    const auto flush_start = std::chrono::steady_clock::now();
    stream.flush();
    const auto flush_end = std::chrono::steady_clock::now();

    if (!stream) {
        return Status::IoError("failed while writing output file: " + output_path.string());
    }

    if (timings != nullptr) {
        timings->header_write_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(header_end - header_start).count();
        timings->vertex_write_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(vertex_end - vertex_start).count();
        timings->flush_ms = std::chrono::duration_cast<std::chrono::milliseconds>(flush_end - flush_start).count();
        timings->total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(flush_end - total_start).count();
    }

    return Status::Ok();
}

} // namespace gs2pc
