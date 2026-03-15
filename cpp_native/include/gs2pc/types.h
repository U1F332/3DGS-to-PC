#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gs2pc {

enum class StatusCode { kOk = 0, kInvalidArgument, kIoError, kNotImplemented, kRuntimeError };

struct Status {
    StatusCode code = StatusCode::kOk;
    std::string message;

    [[nodiscard]] bool ok() const noexcept {
        return code == StatusCode::kOk;
    }

    static Status Ok() {
        return {};
    }

    static Status InvalidArgument(std::string text) {
        return {StatusCode::kInvalidArgument, std::move(text)};
    }

    static Status IoError(std::string text) {
        return {StatusCode::kIoError, std::move(text)};
    }

    static Status NotImplemented(std::string text) {
        return {StatusCode::kNotImplemented, std::move(text)};
    }

    static Status RuntimeError(std::string text) {
        return {StatusCode::kRuntimeError, std::move(text)};
    }
};

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

using Matrix4f = std::array<float, 16>;

enum class InputFileKind { kUnknown = 0, kPointCloudPly, kGaussianPly, kSplat };

inline const char* ToString(InputFileKind kind) noexcept {
    switch (kind) {
    case InputFileKind::kPointCloudPly:
        return "point-cloud-ply";
    case InputFileKind::kGaussianPly:
        return "gaussian-ply";
    case InputFileKind::kSplat:
        return "splat";
    default:
        return "unknown";
    }
}

struct Gaussian {
    Vec3f position;
    Vec3f scale;
    Vec3f normal;
    std::array<float, 4> rotation = {1.0f, 0.0f, 0.0f, 0.0f};
    float opacity = 1.0f;
    std::array<float, 3> color = {1.0f, 1.0f, 1.0f};
    std::vector<float> sh_coefficients;
    bool has_normal = false;
};

struct GaussianSet {
    std::filesystem::path source_path;
    InputFileKind source_kind = InputFileKind::kUnknown;
    std::vector<Gaussian> items;
    bool has_normals = false;
    bool has_sh_coefficients = false;
    bool has_gaussian_attributes = false;

    [[nodiscard]] bool empty() const noexcept {
        return items.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return items.size();
    }
};

struct PointVertex {
    Vec3f position;
    std::array<float, 3> color = {1.0f, 1.0f, 1.0f};
    Vec3f normal;
    bool has_normal = false;
};

struct PointCloud {
    std::vector<PointVertex> points;
};

struct RenderStats {
    std::uint32_t visible_gaussian_count = 0;
    std::uint32_t contributed_gaussian_count = 0;
    std::uint32_t exported_point_count = 0;
};

} // namespace gs2pc
