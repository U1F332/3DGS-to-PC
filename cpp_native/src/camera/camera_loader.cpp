#include "gs2pc/camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace gs2pc {
namespace {

struct ColmapIntrinsics {
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    double fx = 0.0;
    double fy = 0.0;
};

struct ColmapImagePose {
    std::string name;
    int camera_id = -1;
    std::array<double, 4> qvec = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> tvec = {0.0, 0.0, 0.0};
};

float FocalToFov(float focal, float pixels) {
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

Matrix4f Identity4() {
    return {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}

std::array<std::array<double, 3>, 3> QvecToRotmat(const std::array<double, 4>& q) {
    return {{{{1.0 - 2.0 * q[2] * q[2] - 2.0 * q[3] * q[3], 2.0 * q[1] * q[2] - 2.0 * q[0] * q[3],
               2.0 * q[3] * q[1] + 2.0 * q[0] * q[2]}},
             {{2.0 * q[1] * q[2] + 2.0 * q[0] * q[3], 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[3] * q[3],
               2.0 * q[2] * q[3] - 2.0 * q[0] * q[1]}},
             {{2.0 * q[3] * q[1] - 2.0 * q[0] * q[2], 2.0 * q[2] * q[3] + 2.0 * q[0] * q[1],
               1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2]}}}};
}

std::array<std::array<double, 4>, 4> InvertRigid(const std::array<std::array<double, 4>, 4>& m) {
    std::array<std::array<double, 4>, 4> inv = {{{{m[0][0], m[1][0], m[2][0], 0.0}},
                                                 {{m[0][1], m[1][1], m[2][1], 0.0}},
                                                 {{m[0][2], m[1][2], m[2][2], 0.0}},
                                                 {{0.0, 0.0, 0.0, 1.0}}}};

    inv[0][3] = -(inv[0][0] * m[0][3] + inv[0][1] * m[1][3] + inv[0][2] * m[2][3]);
    inv[1][3] = -(inv[1][0] * m[0][3] + inv[1][1] * m[1][3] + inv[1][2] * m[2][3]);
    inv[2][3] = -(inv[2][0] * m[0][3] + inv[2][1] * m[1][3] + inv[2][2] * m[2][3]);

    return inv;
}

std::array<std::array<double, 4>, 4> Multiply4(const std::array<std::array<double, 4>, 4>& a,
                                               const std::array<std::array<double, 4>, 4>& b) {
    std::array<std::array<double, 4>, 4> out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += a[r][k] * b[k][c];
            }
            out[r][c] = sum;
        }
    }
    return out;
}

Matrix4f ToRowMajorFloat(const std::array<std::array<double, 4>, 4>& m) {
    Matrix4f out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r * 4 + c] = static_cast<float>(m[r][c]);
        }
    }
    return out;
}

Matrix4f TransposeRowMajor(const Matrix4f& m) {
    Matrix4f out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r * 4 + c] = m[c * 4 + r];
        }
    }
    return out;
}

Matrix4f MultiplyRowMajor(const Matrix4f& a, const Matrix4f& b) {
    Matrix4f out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = sum;
        }
    }
    return out;
}

Matrix4f BuildProjection(float znear, float zfar, float fov_x, float fov_y) {
    const float tan_half_fov_y = std::tan(fov_y * 0.5f);
    const float tan_half_fov_x = std::tan(fov_x * 0.5f);

    const float top = tan_half_fov_y * znear;
    const float bottom = -top;
    const float right = tan_half_fov_x * znear;
    const float left = -right;

    Matrix4f p{};
    p.fill(0.0f);

    const float z_sign = 1.0f;
    p[0 * 4 + 0] = 2.0f * znear / (right - left);
    p[1 * 4 + 1] = 2.0f * znear / (top - bottom);
    p[0 * 4 + 2] = (right + left) / (right - left);
    p[1 * 4 + 2] = (top + bottom) / (top - bottom);
    p[3 * 4 + 2] = z_sign;
    p[2 * 4 + 2] = z_sign * zfar / (zfar - znear);
    p[2 * 4 + 3] = -(zfar * znear) / (zfar - znear);

    return p;
}

template <typename T> Status ReadBinary(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        return Status::IoError("failed reading COLMAP binary file");
    }
    return Status::Ok();
}

int CameraModelParamCount(int model_id) {
    switch (model_id) {
    case 0:
        return 3; // SIMPLE_PINHOLE
    case 1:
        return 4; // PINHOLE
    case 2:
        return 4; // SIMPLE_RADIAL
    case 3:
        return 5; // RADIAL
    case 4:
        return 8; // OPENCV
    case 5:
        return 8; // OPENCV_FISHEYE
    case 6:
        return 12; // FULL_OPENCV
    case 7:
        return 5; // FOV
    case 8:
        return 4; // SIMPLE_RADIAL_FISHEYE
    case 9:
        return 5; // RADIAL_FISHEYE
    case 10:
        return 12; // THIN_PRISM_FISHEYE
    default:
        return -1;
    }
}

bool DecodeIntrinsicsFromParams(int model_id, const std::vector<double>& params, ColmapIntrinsics& intr) {
    if (model_id == 1 && params.size() >= 4) {
        intr.fx = params[0];
        intr.fy = params[1];
        return true;
    }
    if ((model_id == 0 || model_id == 2 || model_id == 3 || model_id == 8 || model_id == 9 || model_id == 7) &&
        params.size() >= 1) {
        intr.fx = params[0];
        intr.fy = params[0];
        return true;
    }
    if ((model_id == 4 || model_id == 5 || model_id == 6 || model_id == 10) && params.size() >= 4) {
        intr.fx = params[0];
        intr.fy = params[1];
        return true;
    }
    return false;
}

Status LoadColmapBinIntrinsics(const std::filesystem::path& path,
                               std::unordered_map<int, ColmapIntrinsics>& intrinsics) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::IoError("failed to open cameras.bin: " + path.string());
    }

    std::uint64_t num_cameras = 0;
    if (const auto s = ReadBinary(stream, num_cameras); !s.ok()) {
        return s;
    }

    for (std::uint64_t i = 0; i < num_cameras; ++i) {
        std::int32_t camera_id = 0;
        std::int32_t model_id = 0;
        std::uint64_t width = 0;
        std::uint64_t height = 0;
        if (const auto s = ReadBinary(stream, camera_id); !s.ok()) {
            return s;
        }
        if (const auto s = ReadBinary(stream, model_id); !s.ok()) {
            return s;
        }
        if (const auto s = ReadBinary(stream, width); !s.ok()) {
            return s;
        }
        if (const auto s = ReadBinary(stream, height); !s.ok()) {
            return s;
        }

        const int param_count = CameraModelParamCount(model_id);
        if (param_count <= 0) {
            return Status::InvalidArgument("unsupported COLMAP camera model id in cameras.bin: " +
                                           std::to_string(model_id));
        }

        std::vector<double> params(static_cast<std::size_t>(param_count));
        stream.read(reinterpret_cast<char*>(params.data()),
                    static_cast<std::streamsize>(params.size() * sizeof(double)));
        if (!stream) {
            return Status::IoError("failed reading camera params from cameras.bin");
        }

        ColmapIntrinsics intr;
        intr.width = width;
        intr.height = height;
        if (!DecodeIntrinsicsFromParams(model_id, params, intr)) {
            return Status::InvalidArgument("failed to decode camera intrinsics from cameras.bin");
        }
        intrinsics[camera_id] = intr;
    }

    return Status::Ok();
}

Status LoadColmapTxtIntrinsics(const std::filesystem::path& path,
                               std::unordered_map<int, ColmapIntrinsics>& intrinsics) {
    std::ifstream stream(path);
    if (!stream) {
        return Status::IoError("failed to open cameras.txt: " + path.string());
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int camera_id = 0;
        std::string model;
        std::uint64_t width = 0;
        std::uint64_t height = 0;
        if (!(iss >> camera_id >> model >> width >> height)) {
            continue;
        }

        std::vector<double> params;
        double p = 0.0;
        while (iss >> p) {
            params.push_back(p);
        }

        ColmapIntrinsics intr;
        intr.width = width;
        intr.height = height;

        if (model == "PINHOLE" && params.size() >= 4) {
            intr.fx = params[0];
            intr.fy = params[1];
        } else if (!params.empty()) {
            intr.fx = params[0];
            intr.fy = params.size() > 1 ? params[1] : params[0];
        } else {
            continue;
        }

        intrinsics[camera_id] = intr;
    }

    return Status::Ok();
}

Status LoadColmapBinImages(const std::filesystem::path& path, int skip_rate, std::vector<ColmapImagePose>& poses) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::IoError("failed to open images.bin: " + path.string());
    }

    std::uint64_t num_images = 0;
    if (const auto s = ReadBinary(stream, num_images); !s.ok()) {
        return s;
    }

    for (std::uint64_t i = 0; i < num_images; ++i) {
        std::int32_t image_id = 0;
        (void)image_id;

        ColmapImagePose pose;
        if (const auto s = ReadBinary(stream, image_id); !s.ok()) {
            return s;
        }
        for (double& q : pose.qvec) {
            if (const auto s = ReadBinary(stream, q); !s.ok()) {
                return s;
            }
        }
        for (double& t : pose.tvec) {
            if (const auto s = ReadBinary(stream, t); !s.ok()) {
                return s;
            }
        }
        if (const auto s = ReadBinary(stream, pose.camera_id); !s.ok()) {
            return s;
        }

        std::string image_name;
        char ch = '\0';
        do {
            if (const auto s = ReadBinary(stream, ch); !s.ok()) {
                return s;
            }
            if (ch != '\0') {
                image_name.push_back(ch);
            }
        } while (ch != '\0');

        pose.name = std::filesystem::path(image_name).stem().string();

        std::uint64_t num_points2d = 0;
        if (const auto s = ReadBinary(stream, num_points2d); !s.ok()) {
            return s;
        }

        stream.seekg(static_cast<std::streamoff>(num_points2d * (sizeof(double) * 2 + sizeof(std::int64_t))),
                     std::ios::cur);
        if (!stream) {
            return Status::IoError("failed skipping points2D in images.bin");
        }

        if (static_cast<int>(i % static_cast<std::uint64_t>(skip_rate + 1)) == 0) {
            poses.push_back(std::move(pose));
        }
    }

    return Status::Ok();
}

Status LoadColmapTxtImages(const std::filesystem::path& path, int skip_rate, std::vector<ColmapImagePose>& poses) {
    std::ifstream stream(path);
    if (!stream) {
        return Status::IoError("failed to open images.txt: " + path.string());
    }

    std::string line;
    int record_index = 0;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int image_id = 0;
        ColmapImagePose pose;
        std::string image_name;
        if (!(iss >> image_id >> pose.qvec[0] >> pose.qvec[1] >> pose.qvec[2] >> pose.qvec[3] >> pose.tvec[0] >>
              pose.tvec[1] >> pose.tvec[2] >> pose.camera_id >> image_name)) {
            continue;
        }

        pose.name = std::filesystem::path(image_name).stem().string();

        std::string points_line;
        std::getline(stream, points_line);

        if (record_index % (skip_rate + 1) == 0) {
            poses.push_back(std::move(pose));
        }
        ++record_index;
    }

    return Status::Ok();
}

std::optional<std::filesystem::path> ResolveColmapDir(const std::filesystem::path& path) {
    if (std::filesystem::is_regular_file(path)) {
        return std::nullopt;
    }

    if (std::filesystem::exists(path / "images.bin") || std::filesystem::exists(path / "images.txt")) {
        return path;
    }

    const auto sparse = path / "sparse" / "0";
    if (std::filesystem::exists(sparse / "images.bin") || std::filesystem::exists(sparse / "images.txt")) {
        return sparse;
    }

    return std::nullopt;
}

CameraState BuildCameraState(const ColmapIntrinsics& intr, const ColmapImagePose& pose,
                             const ConversionConfig& config) {
    CameraState state;
    state.render = config.render;

    float scale = 1.0f;
    if (config.colour_resolution.has_value() && intr.width > 0) {
        scale = static_cast<float>(*config.colour_resolution) / static_cast<float>(intr.width);
    }

    const int scaled_width = std::max(1, static_cast<int>(std::lround(static_cast<double>(intr.width) * scale)));
    const int scaled_height = std::max(1, static_cast<int>(std::lround(static_cast<double>(intr.height) * scale)));

    state.render.image_width = scaled_width;
    state.render.image_height = scaled_height;

    const float width = static_cast<float>(scaled_width);
    const float height = static_cast<float>(scaled_height);
    const float fx = static_cast<float>(intr.fx) * scale;
    const float fy = static_cast<float>(intr.fy) * scale;

    const float fov_x = FocalToFov(fx, width);
    const float fov_y = FocalToFov(fy, height);

    state.intrinsics.tan_fov_x = std::tan(fov_x * 0.5f);
    state.intrinsics.tan_fov_y = std::tan(fov_y * 0.5f);

    std::array<double, 4> neg_q = {-pose.qvec[0], -pose.qvec[1], -pose.qvec[2], -pose.qvec[3]};
    const auto r = QvecToRotmat(neg_q);

    std::array<std::array<double, 4>, 4> w2c = {{{{r[0][0], r[0][1], r[0][2], pose.tvec[0]}},
                                                 {{r[1][0], r[1][1], r[1][2], pose.tvec[1]}},
                                                 {{r[2][0], r[2][1], r[2][2], pose.tvec[2]}},
                                                 {{0.0, 0.0, 0.0, 1.0}}}};

    auto c2w = InvertRigid(w2c);

    const std::array<std::array<double, 4>, 4> flip = {
        {{{1.0, 0.0, 0.0, 0.0}}, {{0.0, -1.0, 0.0, 0.0}}, {{0.0, 0.0, -1.0, 0.0}}, {{0.0, 0.0, 0.0, 1.0}}}};

    c2w = Multiply4(c2w, flip);

    for (int r_i = 0; r_i < 4; ++r_i) {
        c2w[r_i][1] *= -1.0;
        c2w[r_i][2] *= -1.0;
    }

    const auto view = TransposeRowMajor(ToRowMajorFloat(InvertRigid(c2w)));
    const auto proj = TransposeRowMajor(BuildProjection(10.0f, 100.0f, fov_x, fov_y));

    state.pose.view_matrix = view;
    state.pose.proj_matrix = MultiplyRowMajor(view, proj);
    state.pose.camera_pos = {static_cast<float>(c2w[0][3]), static_cast<float>(c2w[1][3]),
                             static_cast<float>(c2w[2][3])};

    return state;
}

} // namespace

Status LoadCameraFrames(const std::filesystem::path& transform_path, const ConversionConfig& config,
                        std::vector<CameraFrame>& frames) {
    frames.clear();

    const auto colmap_dir = ResolveColmapDir(transform_path);
    if (!colmap_dir.has_value()) {
        return Status::InvalidArgument("unsupported transform path (expected COLMAP folder): " +
                                       transform_path.string());
    }

    std::unordered_map<int, ColmapIntrinsics> intrinsics;
    std::vector<ColmapImagePose> poses;

    if (std::filesystem::exists(*colmap_dir / "cameras.bin") && std::filesystem::exists(*colmap_dir / "images.bin")) {
        if (const auto s = LoadColmapBinIntrinsics(*colmap_dir / "cameras.bin", intrinsics); !s.ok()) {
            return s;
        }
        if (const auto s = LoadColmapBinImages(*colmap_dir / "images.bin", config.camera_skip_rate, poses); !s.ok()) {
            return s;
        }
    } else if (std::filesystem::exists(*colmap_dir / "cameras.txt") &&
               std::filesystem::exists(*colmap_dir / "images.txt")) {
        if (const auto s = LoadColmapTxtIntrinsics(*colmap_dir / "cameras.txt", intrinsics); !s.ok()) {
            return s;
        }
        if (const auto s = LoadColmapTxtImages(*colmap_dir / "images.txt", config.camera_skip_rate, poses); !s.ok()) {
            return s;
        }
    } else {
        return Status::InvalidArgument("COLMAP directory must contain cameras/images in .bin or .txt: " +
                                       colmap_dir->string());
    }

    if (poses.empty()) {
        return Status::InvalidArgument("no camera poses found in transform path: " + colmap_dir->string());
    }

    frames.reserve(poses.size());
    for (const auto& pose : poses) {
        const auto it = intrinsics.find(pose.camera_id);
        if (it == intrinsics.end()) {
            continue;
        }

        CameraFrame frame;
        frame.name = pose.name;
        frame.state = BuildCameraState(it->second, pose, config);
        frames.push_back(std::move(frame));
    }

    if (frames.empty()) {
        return Status::InvalidArgument("no valid camera frames constructed from transform path: " +
                                       colmap_dir->string());
    }

    return Status::Ok();
}

} // namespace gs2pc
