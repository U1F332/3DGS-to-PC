#pragma once

#include "gs2pc/types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gs2pc {

struct CameraIntrinsics {
    float tan_fov_x = 1.0f;
    float tan_fov_y = 1.0f;
};

struct CameraPose {
    Matrix4f view_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    Matrix4f proj_matrix = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    Vec3f camera_pos;
};

struct RenderMask {
    std::vector<std::int32_t> pixel_mask;
};

struct RenderConfig {
    int image_width = 1280;
    int image_height = 720;
    bool prefiltered = false;
    bool antialiasing = false;
    bool debug = false;
};

struct GaussConfig {
    float scale_modifier = 1.0f;
};

struct PointCloudConfig {
    float visible_threshold = 0.05f;
    std::optional<float> surface_distance_std;
    bool enable_surface_distance = false;
};

struct ConversionConfig {
    std::filesystem::path input_path;
    std::filesystem::path output_path = "3dgs_pc.ply";
    std::optional<std::filesystem::path> transform_path;
    std::optional<std::filesystem::path> mask_path;

    std::string renderer_type = "cuda";
    int num_points = 10000000;
    bool exact_num_points = false;
    bool prioritise_visible_gaussians = true;
    int camera_skip_rate = 0;
    bool render_colours = true;
    std::optional<int> colour_resolution = 1280;
    bool calculate_normals = true;
    bool clean_pointcloud = false;
    bool generate_mesh = false;
    int poisson_depth = 10;
    int laplacian_iterations = 10;
    std::filesystem::path mesh_output_path = "3dgs_mesh.ply";
    std::optional<std::array<float, 3>> bounding_box_min;
    std::optional<std::array<float, 3>> bounding_box_max;
    float mahalanobis_distance_std = 2.0f;
    float min_opacity = 0.0f;
    float cull_gaussian_sizes = 0.0f;
    int max_sh_degree = 3;
    bool quiet = false;

    CameraIntrinsics camera_intrinsics;
    CameraPose camera_pose;
    RenderConfig render;
    GaussConfig gauss;
    PointCloudConfig point_cloud;

    bool run_mark_visible = false;
    bool run_forward = false;
    bool export_visible_only = false;

    bool show_help = false;
    bool show_version = false;
};

using CliOptions = ConversionConfig;

} // namespace gs2pc
