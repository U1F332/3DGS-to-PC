#include "gs2pc/camera.h"
#include "gs2pc/converter.h"
#include "gs2pc/exporter.h"
#include "gs2pc/io.h"
#include "gs2pc/raster.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShC1 = 0.4886025119029199f;
constexpr std::array<float, 5> kShC2 = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                        -1.0925484305920792f, 0.5462742152960396f};
constexpr std::array<float, 7> kShC3 = {-0.5900435899266435f, 2.890611442640554f,  -0.4570457994644658f,
                                        0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                                        -0.5900435899266435f};
constexpr std::array<float, 9> kShC4 = {2.5033429417967046f,  -1.7701307697799304f, 0.9461746957575601f,
                                        -0.6690465435572892f, 0.10578554691520431f, -0.6690465435572892f,
                                        0.47308734787878004f, -1.7701307697799304f, 0.6258357354491761f};

gs2pc::GaussianSet FilterVisibleGaussians(const gs2pc::GaussianSet& source, const std::vector<bool>& present) {
    gs2pc::GaussianSet filtered;
    filtered.source_path = source.source_path;
    filtered.source_kind = source.source_kind;
    filtered.has_normals = source.has_normals;
    filtered.has_sh_coefficients = source.has_sh_coefficients;
    filtered.has_gaussian_attributes = source.has_gaussian_attributes;
    filtered.items.reserve(source.items.size());

    for (std::size_t i = 0; i < source.items.size() && i < present.size(); ++i) {
        if (present[i]) {
            filtered.items.push_back(source.items[i]);
        }
    }

    return filtered;
}

gs2pc::GaussianSet FilterGaussiansByMask(const gs2pc::GaussianSet& source, const std::vector<bool>& keep_mask) {
    gs2pc::GaussianSet filtered;
    filtered.source_path = source.source_path;
    filtered.source_kind = source.source_kind;
    filtered.has_normals = source.has_normals;
    filtered.has_sh_coefficients = source.has_sh_coefficients;
    filtered.has_gaussian_attributes = source.has_gaussian_attributes;
    filtered.items.reserve(source.items.size());

    for (std::size_t i = 0; i < source.items.size() && i < keep_mask.size(); ++i) {
        if (keep_mask[i]) {
            filtered.items.push_back(source.items[i]);
        }
    }

    return filtered;
}

std::size_t CountVisible(const std::vector<bool>& mask) {
    std::size_t count = 0;
    for (const bool v : mask) {
        count += v ? 1u : 0u;
    }
    return count;
}

float SafeChannel(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

float Dot(const gs2pc::Vec3f& a, const gs2pc::Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

gs2pc::Vec3f Normalize(const gs2pc::Vec3f& v) {
    const float n = std::sqrt(std::max(Dot(v, v), 0.0f));
    if (n <= std::numeric_limits<float>::epsilon()) {
        return {0.0f, 0.0f, 1.0f};
    }
    return {v.x / n, v.y / n, v.z / n};
}

gs2pc::Vec3f DirectionToCamera(const gs2pc::Gaussian& gaussian, const gs2pc::CameraState& camera) {
    return Normalize({gaussian.position.x - camera.pose.camera_pos.x, gaussian.position.y - camera.pose.camera_pos.y,
                      gaussian.position.z - camera.pose.camera_pos.z});
}

std::array<float, 3> EvalShColour(const gs2pc::Gaussian& gaussian, int max_degree, const gs2pc::Vec3f& dir) {
    if (gaussian.sh_coefficients.size() < 3) {
        return gaussian.color;
    }

    const int degree = std::clamp(max_degree, 0, 4);
    const int coeff_count = (degree + 1) * (degree + 1);
    const int required_coeffs = coeff_count * 3;
    if (static_cast<int>(gaussian.sh_coefficients.size()) < required_coeffs) {
        return gaussian.color;
    }

    auto eval_channel = [&](int channel) {
        float result = 0.0f;
        const auto coeff_at = [&](int coeff_index) -> float {
            if (coeff_index == 0) {
                return gaussian.sh_coefficients[static_cast<std::size_t>(channel)];
            }

            const std::size_t base = 3 + static_cast<std::size_t>(coeff_index - 1) * 3;
            const std::size_t flat_index = base + static_cast<std::size_t>(channel);
            if (flat_index >= gaussian.sh_coefficients.size()) {
                return 0.0f;
            }
            return gaussian.sh_coefficients[flat_index];
        };

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        result = kShC0 * coeff_at(0);

        if (degree > 0) {
            result += -kShC1 * y * coeff_at(1);
            result += kShC1 * z * coeff_at(2);
            result += -kShC1 * x * coeff_at(3);
        }
        if (degree > 1) {
            result += kShC2[0] * xy * coeff_at(4);
            result += kShC2[1] * yz * coeff_at(5);
            result += kShC2[2] * (2.0f * zz - xx - yy) * coeff_at(6);
            result += kShC2[3] * xz * coeff_at(7);
            result += kShC2[4] * (xx - yy) * coeff_at(8);
        }
        if (degree > 2) {
            result += kShC3[0] * y * (3.0f * xx - yy) * coeff_at(9);
            result += kShC3[1] * xy * z * coeff_at(10);
            result += kShC3[2] * y * (4.0f * zz - xx - yy) * coeff_at(11);
            result += kShC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * coeff_at(12);
            result += kShC3[4] * x * (4.0f * zz - xx - yy) * coeff_at(13);
            result += kShC3[5] * z * (xx - yy) * coeff_at(14);
            result += kShC3[6] * x * (xx - 3.0f * yy) * coeff_at(15);
        }
        if (degree > 3) {
            result += kShC4[0] * xy * (xx - yy) * coeff_at(16);
            result += kShC4[1] * yz * (3.0f * xx - yy) * coeff_at(17);
            result += kShC4[2] * xy * (7.0f * zz - 1.0f) * coeff_at(18);
            result += kShC4[3] * yz * (7.0f * zz - 3.0f) * coeff_at(19);
            result += kShC4[4] * (zz * (35.0f * zz - 30.0f) + 3.0f) * coeff_at(20);
            result += kShC4[5] * xz * (7.0f * zz - 3.0f) * coeff_at(21);
            result += kShC4[6] * (xx - yy) * (7.0f * zz - 1.0f) * coeff_at(22);
            result += kShC4[7] * xz * (xx - 3.0f * yy) * coeff_at(23);
            result += kShC4[8] * (xx * (xx - 3.0f * yy) - yy * (3.0f * xx - yy)) * coeff_at(24);
        }

        return SafeChannel(result + 0.5f);
    };

    return {eval_channel(0), eval_channel(1), eval_channel(2)};
}

void ApplyShColourFromCameras(gs2pc::GaussianSet& gaussians, const std::vector<gs2pc::CameraFrame>& camera_frames,
                              int max_sh_degree) {
    if (gaussians.items.empty() || camera_frames.empty() || !gaussians.has_sh_coefficients) {
        return;
    }

    for (auto& gaussian : gaussians.items) {
        if (gaussian.sh_coefficients.size() < 3) {
            continue;
        }

        std::array<float, 3> accum = {0.0f, 0.0f, 0.0f};
        for (const auto& frame : camera_frames) {
            const auto dir = DirectionToCamera(gaussian, frame.state);
            const auto c = EvalShColour(gaussian, max_sh_degree, dir);
            accum[0] += c[0];
            accum[1] += c[1];
            accum[2] += c[2];
        }

        const float inv = 1.0f / static_cast<float>(camera_frames.size());
        gaussian.color = {SafeChannel(accum[0] * inv), SafeChannel(accum[1] * inv), SafeChannel(accum[2] * inv)};
    }
}

} // namespace

int main(int argc, const char* const argv[]) {
    gs2pc::CliOptions options;
    const auto parse_status = gs2pc::ParseCommandLine(argc, argv, options);
    if (!parse_status.ok()) {
        std::cerr << "Error: " << parse_status.message << "\n\n";
        std::cerr << gs2pc::BuildHelpText();
        return 1;
    }

    if (options.show_help) {
        std::cout << gs2pc::BuildHelpText();
        return 0;
    }

    if (!options.quiet) {
        std::cout << "Loading gaussians from: " << options.input_path.string() << '\n';
    }

    gs2pc::GaussianSet gaussians;
    const auto load_status = gs2pc::LoadGaussians(options.input_path, gaussians);
    if (!load_status.ok()) {
        std::cerr << "Error: " << load_status.message << '\n';
        return 1;
    }

    if (!options.quiet) {
        std::cout << "Detected input kind: " << gs2pc::ToString(gaussians.source_kind) << '\n';
        std::cout << "Has normals: " << (gaussians.has_normals ? "yes" : "no") << '\n';
        std::cout << "Has SH coefficients: " << (gaussians.has_sh_coefficients ? "yes" : "no") << '\n';
        std::cout << "Has Gaussian attributes: " << (gaussians.has_gaussian_attributes ? "yes" : "no") << '\n';
    }

    std::vector<gs2pc::CameraFrame> camera_frames;
    if (options.transform_path.has_value()) {
        const auto cam_status = gs2pc::LoadCameraFrames(*options.transform_path, options, camera_frames);
        if (!cam_status.ok()) {
            std::cerr << "Error: " << cam_status.message << '\n';
            return 1;
        }

        if (!options.quiet) {
            std::cout << "Loaded camera frames: " << camera_frames.size() << " from "
                      << options.transform_path->string() << '\n';
        }

        if (!camera_frames.empty() && gaussians.has_sh_coefficients) {
            ApplyShColourFromCameras(gaussians, camera_frames, options.max_sh_degree);
            if (!options.quiet) {
                std::cout << "Applied camera-aware SH colour averaging using loaded camera frames\n";
            }
        }
    }

    gs2pc::GaussianSet working_gaussians = gaussians;

    if (options.run_mark_visible) {
        if (!gs2pc::HasCudaRasterizer()) {
            std::cerr << "Error: this build does not include CUDA raster support; reconfigure with "
                         "GS2PC_ENABLE_CUDA_RASTER=ON\n";
            return 1;
        }

        std::vector<bool> visible_mask(working_gaussians.size(), false);

        if (!camera_frames.empty()) {
            for (const auto& frame : camera_frames) {
                std::vector<bool> current_visible;
                gs2pc::RasterFrameInputs inputs;
                inputs.gaussians = &working_gaussians;
                inputs.camera = &frame.state;

                const auto visible_status = gs2pc::MarkVisible(inputs, current_visible);
                if (!visible_status.ok()) {
                    std::cerr << "Error: " << visible_status.message << '\n';
                    return 1;
                }

                for (std::size_t g = 0; g < visible_mask.size() && g < current_visible.size(); ++g) {
                    visible_mask[g] = visible_mask[g] || current_visible[g];
                }
            }
        } else {
            gs2pc::CameraState camera_state;
            camera_state.intrinsics = options.camera_intrinsics;
            camera_state.pose = options.camera_pose;
            camera_state.render = options.render;

            gs2pc::RasterFrameInputs inputs;
            inputs.gaussians = &working_gaussians;
            inputs.camera = &camera_state;

            const auto visible_status = gs2pc::MarkVisible(inputs, visible_mask);
            if (!visible_status.ok()) {
                std::cerr << "Error: " << visible_status.message << '\n';
                return 1;
            }
        }

        const std::size_t visible_count = CountVisible(visible_mask);
        if (!options.quiet) {
            std::cout << "markVisible completed. Visible gaussians: " << visible_count << " / "
                      << working_gaussians.size() << '\n';
        }

        if (options.export_visible_only) {
            working_gaussians = FilterVisibleGaussians(working_gaussians, visible_mask);

            if (!options.quiet) {
                std::cout << "Exporting visible-only subset: " << working_gaussians.size() << " gaussians\n";
            }
        }
    }

    if (options.run_forward) {
        if (!gs2pc::HasCudaRasterizer()) {
            std::cerr << "Error: this build does not include CUDA raster support; reconfigure with "
                         "GS2PC_ENABLE_CUDA_RASTER=ON\n";
            return 1;
        }

        std::vector<gs2pc::CameraState> forward_cameras;
        if (!camera_frames.empty()) {
            forward_cameras.reserve(camera_frames.size());
            for (const auto& frame : camera_frames) {
                forward_cameras.push_back(frame.state);
            }
        } else {
            gs2pc::CameraState fallback;
            fallback.intrinsics = options.camera_intrinsics;
            fallback.pose = options.camera_pose;
            fallback.render = options.render;
            forward_cameras.push_back(std::move(fallback));
        }

        std::vector<float> max_contrib(working_gaussians.size(), 0.0f);
        std::vector<float> total_contrib(working_gaussians.size(), 0.0f);
        std::vector<float> min_surface_dist(working_gaussians.size(), std::numeric_limits<float>::max());
        std::vector<std::array<float, 3>> best_colors(working_gaussians.size(), {0.0f, 0.0f, 0.0f});

        std::size_t total_rendered_instances = 0;

        for (const auto& cam : forward_cameras) {
            gs2pc::RasterFrameInputs forward_inputs;
            forward_inputs.gaussians = &working_gaussians;
            forward_inputs.camera = &cam;
            forward_inputs.gauss_config = &options.gauss;
            forward_inputs.sh_degree = options.max_sh_degree;
            forward_inputs.calculate_surface_distance = options.point_cloud.enable_surface_distance;

            gs2pc::RasterFrameOutputs forward_outputs;
            const auto forward_status = gs2pc::RasterizeForward(forward_inputs, forward_outputs);
            if (!forward_status.ok()) {
                std::cerr << "Error: " << forward_status.message << '\n';
                return 1;
            }

            total_rendered_instances += static_cast<std::size_t>(forward_outputs.rendered);

            const int pixel_count = cam.render.image_width * cam.render.image_height;
            for (std::size_t i = 0; i < working_gaussians.size() && i < forward_outputs.gauss_contributions.size();
                 ++i) {
                const float c = forward_outputs.gauss_contributions[i];
                total_contrib[i] += c;

                if (c > max_contrib[i]) {
                    max_contrib[i] = c;

                    if (i < forward_outputs.gauss_pixels.size()) {
                        const int pixel_idx = forward_outputs.gauss_pixels[i];
                        if (pixel_idx >= 0 && pixel_idx < pixel_count) {
                            const int r_idx = pixel_idx;
                            const int g_idx = pixel_count + pixel_idx;
                            const int b_idx = pixel_count * 2 + pixel_idx;
                            if (b_idx < static_cast<int>(forward_outputs.color.size())) {
                                best_colors[i] = {SafeChannel(forward_outputs.color[r_idx]),
                                                  SafeChannel(forward_outputs.color[g_idx]),
                                                  SafeChannel(forward_outputs.color[b_idx])};
                            }
                        }
                    }
                }

                if (options.point_cloud.enable_surface_distance && i < forward_outputs.gauss_surface_distances.size()) {
                    min_surface_dist[i] = std::min(min_surface_dist[i], forward_outputs.gauss_surface_distances[i]);
                }
            }
        }

        for (std::size_t i = 0; i < working_gaussians.size(); ++i) {
            if (max_contrib[i] > 0.0f) {
                working_gaussians.items[i].color = best_colors[i];
            }
        }

        std::vector<bool> keep_mask(working_gaussians.size(), true);

        if (options.point_cloud.visible_threshold > 0.0f && !max_contrib.empty()) {
            for (std::size_t i = 0; i < keep_mask.size(); ++i) {
                keep_mask[i] = keep_mask[i] && (max_contrib[i] > options.point_cloud.visible_threshold);
            }
        }

        if (options.point_cloud.enable_surface_distance && options.point_cloud.surface_distance_std.has_value()) {
            double sum = 0.0;
            std::size_t valid_count = 0;
            for (const float d : min_surface_dist) {
                if (d < std::numeric_limits<float>::max()) {
                    sum += d;
                    ++valid_count;
                }
            }

            if (valid_count > 0) {
                const float mean_dist = static_cast<float>(sum / static_cast<double>(valid_count));
                const float threshold = mean_dist * (*options.point_cloud.surface_distance_std);
                for (std::size_t i = 0; i < keep_mask.size(); ++i) {
                    if (min_surface_dist[i] < std::numeric_limits<float>::max()) {
                        keep_mask[i] = keep_mask[i] && (min_surface_dist[i] < threshold);
                    }
                }
            }
        }

        if (options.prioritise_visible_gaussians || options.point_cloud.visible_threshold > 0.0f ||
            options.point_cloud.enable_surface_distance) {
            const auto before = working_gaussians.size();
            working_gaussians = FilterGaussiansByMask(working_gaussians, keep_mask);
            const auto after = working_gaussians.size();

            if (!options.quiet) {
                std::cout << "forward filtering kept " << after << " / " << before << " gaussians\n";
            }
        }

        if (!options.quiet) {
            float max_value = 0.0f;
            if (!max_contrib.empty()) {
                max_value = *std::max_element(max_contrib.begin(), max_contrib.end());
            }
            const double total_value = std::accumulate(total_contrib.begin(), total_contrib.end(), 0.0);
            std::cout << "forward completed over " << forward_cameras.size()
                      << " camera(s). rendered instances sum: " << total_rendered_instances << '\n';
            std::cout << "forward gauss contribution sum/max: " << total_value << " / " << max_value << '\n';
        }
    }

    gs2pc::PointCloud point_cloud;
    gs2pc::RenderStats stats;
    const auto convert_status = gs2pc::ConvertGaussiansToPointCloud(working_gaussians, options, point_cloud, &stats);
    if (!convert_status.ok()) {
        std::cerr << "Error: " << convert_status.message << '\n';
        return 1;
    }

    const auto export_status = gs2pc::ExportPointCloudPly(point_cloud, options.output_path);
    if (!export_status.ok()) {
        std::cerr << "Error: " << export_status.message << '\n';
        return 1;
    }

    if (!options.quiet) {
        std::cout << "Loaded gaussians: " << gaussians.size() << '\n';
        std::cout << "Exported points: " << stats.exported_point_count << '\n';
        std::cout << "Saved point cloud to: " << options.output_path.string() << '\n';
        std::cout << "Note: native converter now samples points from filtered Gaussians toward --num_points.\n";
    }

    return 0;
}
