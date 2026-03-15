#include "gs2pc/io.h"

#include <array>
#include <charconv>
#include <optional>
#include <sstream>
#include <string_view>

namespace gs2pc {
namespace {

constexpr std::array<std::pair<std::string_view, int>, 6> kColourQualityOptions = {{
    {"tiny", 180},
    {"low", 360},
    {"medium", 720},
    {"high", 1280},
    {"ultra", 1920},
    {"original", -1},
}};

Status RequireValue(int index, int argc, std::string_view option) {
    if (index + 1 >= argc) {
        return Status::InvalidArgument(std::string(option) + " requires a value");
    }
    return Status::Ok();
}

Status RequireValues(int index, int argc, int value_count, std::string_view option) {
    if (index + value_count >= argc) {
        return Status::InvalidArgument(std::string(option) + " requires " + std::to_string(value_count) + " values");
    }
    return Status::Ok();
}

Status ParseInt(std::string_view text, int& value, std::string_view option) {
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return Status::InvalidArgument(std::string(option) + " must be an integer");
    }
    return Status::Ok();
}

Status ParseFloat(std::string_view text, float& value, std::string_view option) {
    try {
        size_t processed = 0;
        value = std::stof(std::string(text), &processed);
        if (processed != text.size()) {
            return Status::InvalidArgument(std::string(option) + " must be a float");
        }
    } catch (...) {
        return Status::InvalidArgument(std::string(option) + " must be a float");
    }
    return Status::Ok();
}

Status ParseVec3(int& i, int argc, const char* const argv[], std::array<float, 3>& out, std::string_view option) {
    if (const auto status = RequireValues(i, argc, 3, option); !status.ok()) {
        return status;
    }

    for (int j = 0; j < 3; ++j) {
        if (const auto status = ParseFloat(argv[++i], out[j], option); !status.ok()) {
            return status;
        }
    }

    return Status::Ok();
}

std::optional<int> LookupColourResolution(std::string_view quality) {
    for (const auto& [name, resolution] : kColourQualityOptions) {
        if (name == quality) {
            if (resolution < 0) {
                return std::nullopt;
            }
            return resolution;
        }
    }
    return 1280;
}

bool IsKnownColourQuality(std::string_view quality) {
    for (const auto& [name, _] : kColourQualityOptions) {
        if (name == quality) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string BuildHelpText() {
    std::ostringstream oss;
    oss << "gs2pc_cli - native 3DGS to point cloud scaffold\n\n";
    oss << "Usage:\n";
    oss << "  gs2pc_cli --input_path <scene.ply|scene.splat> [options]\n\n";
    oss << "Core options:\n";
    oss << "  --help                         Show this help message\n";
    oss << "  --input_path <path>            Input Gaussian scene\n";
    oss << "  --output_path <path>           Output point cloud path (default: 3dgs_pc.ply)\n";
    oss << "  --transform_path <path>        Camera transform source\n";
    oss << "  --mask_path <path>             Mask directory\n";
    oss << "  --renderer_type <cuda|python>  Renderer type placeholder (default: cuda)\n";
    oss << "  --num_points <int>             Target point count (default: 10000000)\n";
    oss << "  --visibility_threshold <float> Visible contribution threshold (default: 0.05)\n";
    oss << "  --surface_distance_std <float> Surface distance culling threshold\n";
    oss << "  --mahalanobis_distance_std <float> Max sampling distance in std space\n";
    oss << "  --min_opacity <float>          Minimum Gaussian opacity to keep\n";
    oss << "  --cull_gaussian_sizes <float>  Cull largest Gaussian ratio [0,1)\n";
    oss << "  --bounding_box_min <x y z>     Bounding-box minimum\n";
    oss << "  --bounding_box_max <x y z>     Bounding-box maximum\n";
    oss << "  --max_sh_degree <int>          Maximum SH degree\n";
    oss << "  --colour_quality <name>        tiny|low|medium|high|ultra|original\n";
    oss << "  --camera_skip_rate <int>       Skip rate for camera list\n";
    oss << "  --image_width <int>            Diagnostic camera width (default: 1280)\n";
    oss << "  --image_height <int>           Diagnostic camera height (default: 720)\n";
    oss << "  --tanfovx <float>              Diagnostic camera tan_fov_x (default: 1.0)\n";
    oss << "  --tanfovy <float>              Diagnostic camera tan_fov_y (default: 1.0)\n";
    oss << "  --exact_num_points             Enable stricter point count matching\n";
    oss << "  --no_prioritise_visible_gaussians  Disable contribution-based prioritisation\n";
    oss << "  --no_calculate_normals         Disable normal estimation\n";
    oss << "  --no_render_colours            Skip colour rendering\n";
    oss << "  --run_mark_visible             Run native CUDA markVisible diagnostics\n";
    oss << "  --run_forward                  Run native CUDA forward diagnostics\n";
    oss << "  --export_visible_only          Export only Gaussians marked visible by the diagnostic camera\n";
    oss << "  --quiet                        Reduce stdout output\n";
    return oss.str();
}

Status ParseCommandLine(int argc, const char* const argv[], CliOptions& options) {
    if (argc <= 1) {
        options.show_help = true;
        return Status::Ok();
    }

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return Status::Ok();
        }
        if (arg == "--input_path") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            options.input_path = argv[++i];
            continue;
        }
        if (arg == "--output_path") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            options.output_path = argv[++i];
            continue;
        }
        if (arg == "--transform_path") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            options.transform_path = std::filesystem::path(argv[++i]);
            continue;
        }
        if (arg == "--mask_path") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            options.mask_path = std::filesystem::path(argv[++i]);
            continue;
        }
        if (arg == "--renderer_type") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            options.renderer_type = argv[++i];
            continue;
        }
        if (arg == "--num_points") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseInt(argv[++i], options.num_points, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--visibility_threshold") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.point_cloud.visible_threshold, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--surface_distance_std") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            float value = 0.0f;
            if (const auto status = ParseFloat(argv[++i], value, arg); !status.ok()) {
                return status;
            }
            options.point_cloud.surface_distance_std = value;
            options.point_cloud.enable_surface_distance = true;
            continue;
        }
        if (arg == "--mahalanobis_distance_std") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.mahalanobis_distance_std, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--min_opacity") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.min_opacity, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--cull_gaussian_sizes") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.cull_gaussian_sizes, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--bounding_box_min") {
            std::array<float, 3> value{};
            if (const auto status = ParseVec3(i, argc, argv, value, arg); !status.ok()) {
                return status;
            }
            options.bounding_box_min = value;
            continue;
        }
        if (arg == "--bounding_box_max") {
            std::array<float, 3> value{};
            if (const auto status = ParseVec3(i, argc, argv, value, arg); !status.ok()) {
                return status;
            }
            options.bounding_box_max = value;
            continue;
        }
        if (arg == "--max_sh_degree") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseInt(argv[++i], options.max_sh_degree, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--colour_quality") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            const std::string_view value = argv[++i];
            if (!IsKnownColourQuality(value)) {
                return Status::InvalidArgument(
                    "--colour_quality must be one of tiny, low, medium, high, ultra, original");
            }
            options.colour_resolution = LookupColourResolution(value);
            continue;
        }
        if (arg == "--camera_skip_rate") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseInt(argv[++i], options.camera_skip_rate, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--image_width") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseInt(argv[++i], options.render.image_width, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--image_height") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseInt(argv[++i], options.render.image_height, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--tanfovx") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.camera_intrinsics.tan_fov_x, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--tanfovy") {
            if (const auto status = RequireValue(i, argc, arg); !status.ok()) {
                return status;
            }
            if (const auto status = ParseFloat(argv[++i], options.camera_intrinsics.tan_fov_y, arg); !status.ok()) {
                return status;
            }
            continue;
        }
        if (arg == "--exact_num_points") {
            options.exact_num_points = true;
            continue;
        }
        if (arg == "--no_prioritise_visible_gaussians") {
            options.prioritise_visible_gaussians = false;
            continue;
        }
        if (arg == "--no_calculate_normals") {
            options.calculate_normals = false;
            continue;
        }
        if (arg == "--no_render_colours") {
            options.render_colours = false;
            continue;
        }
        if (arg == "--run_mark_visible") {
            options.run_mark_visible = true;
            continue;
        }
        if (arg == "--run_forward") {
            options.run_forward = true;
            continue;
        }
        if (arg == "--export_visible_only") {
            options.export_visible_only = true;
            continue;
        }
        if (arg == "--quiet") {
            options.quiet = true;
            continue;
        }

        return Status::InvalidArgument("unknown argument: " + std::string(arg));
    }

    if (options.input_path.empty()) {
        return Status::InvalidArgument("--input_path is required unless --help is used");
    }
    if (options.num_points <= 0) {
        return Status::InvalidArgument("--num_points must be greater than zero");
    }
    if (options.camera_skip_rate < 0) {
        return Status::InvalidArgument("--camera_skip_rate must be greater than or equal to zero");
    }
    if (options.render.image_width <= 0 || options.render.image_height <= 0) {
        return Status::InvalidArgument("--image_width and --image_height must be greater than zero");
    }
    if (options.camera_intrinsics.tan_fov_x <= 0.0f || options.camera_intrinsics.tan_fov_y <= 0.0f) {
        return Status::InvalidArgument("--tanfovx and --tanfovy must be greater than zero");
    }
    if (options.point_cloud.visible_threshold < 0.0f || options.point_cloud.visible_threshold > 1.0f) {
        return Status::InvalidArgument("--visibility_threshold must be between 0 and 1");
    }
    if (options.point_cloud.surface_distance_std.has_value() && *options.point_cloud.surface_distance_std <= 0.0f) {
        return Status::InvalidArgument("--surface_distance_std must be greater than zero");
    }
    if (options.mahalanobis_distance_std <= 0.0f) {
        return Status::InvalidArgument("--mahalanobis_distance_std must be greater than zero");
    }
    if (options.min_opacity < 0.0f || options.min_opacity > 1.0f) {
        return Status::InvalidArgument("--min_opacity must be between 0 and 1");
    }
    if (options.cull_gaussian_sizes < 0.0f || options.cull_gaussian_sizes >= 1.0f) {
        return Status::InvalidArgument("--cull_gaussian_sizes must be in [0, 1)");
    }
    if (options.max_sh_degree < 0) {
        return Status::InvalidArgument("--max_sh_degree must be greater than or equal to zero");
    }
    if (options.export_visible_only && !options.run_mark_visible) {
        return Status::InvalidArgument("--export_visible_only requires --run_mark_visible");
    }

    return Status::Ok();
}

} // namespace gs2pc
