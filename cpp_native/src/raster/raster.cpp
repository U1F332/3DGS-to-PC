#include "gs2pc/raster.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(GS2PC_HAS_CUDA_RASTER)
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#endif

namespace gs2pc {
namespace {

#if defined(GS2PC_HAS_CUDA_RASTER)
Status MakeCudaErrorStatus(const std::string& action, cudaError_t error) {
    return Status::RuntimeError(action + ": " + cudaGetErrorString(error));
}

template <typename T> Status CopyHostToDevice(const T* host_data, std::size_t count, T*& device_data) {
    device_data = nullptr;
    if (count == 0) {
        return Status::Ok();
    }

    const auto alloc_error = cudaMalloc(reinterpret_cast<void**>(&device_data), count * sizeof(T));
    if (alloc_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMalloc failed", alloc_error);
    }

    const auto copy_error = cudaMemcpy(device_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) {
        cudaFree(device_data);
        device_data = nullptr;
        return MakeCudaErrorStatus("cudaMemcpy host->device failed", copy_error);
    }

    return Status::Ok();
}

Status EnsureCudaDevice() {
    int device_count = 0;
    const auto count_error = cudaGetDeviceCount(&device_count);
    if (count_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaGetDeviceCount failed", count_error);
    }
    if (device_count <= 0) {
        return Status::RuntimeError("no CUDA device available");
    }

    const auto set_device_error = cudaSetDevice(0);
    if (set_device_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaSetDevice failed", set_device_error);
    }

    return Status::Ok();
}
#endif

} // namespace

bool HasCudaRasterizer() noexcept {
#if defined(GS2PC_HAS_CUDA_RASTER)
    return true;
#else
    return false;
#endif
}

Status MarkVisible(const RasterFrameInputs& inputs, std::vector<bool>& present) {
    present.clear();

    if (inputs.gaussians == nullptr) {
        return Status::InvalidArgument("markVisible requires a non-null gaussian set");
    }
    if (inputs.camera == nullptr) {
        return Status::InvalidArgument("markVisible requires a non-null camera state");
    }

    const auto& gaussians = *inputs.gaussians;
    const auto& camera = *inputs.camera;

    if (const auto camera_status = ValidateCameraState(camera); !camera_status.ok()) {
        return camera_status;
    }

    present.assign(gaussians.size(), false);
    if (gaussians.empty()) {
        return Status::Ok();
    }

#if !defined(GS2PC_HAS_CUDA_RASTER)
    return Status::NotImplemented("CUDA rasterizer support is disabled in this build");
#else
    if (const auto device_status = EnsureCudaDevice(); !device_status.ok()) {
        return device_status;
    }

    std::vector<float> means3d;
    means3d.reserve(gaussians.size() * 3);
    for (const auto& gaussian : gaussians.items) {
        means3d.push_back(gaussian.position.x);
        means3d.push_back(gaussian.position.y);
        means3d.push_back(gaussian.position.z);
    }

    float* d_means3d = nullptr;
    float* d_view = nullptr;
    float* d_proj = nullptr;
    bool* d_present = nullptr;

    const auto free_all = [&]() {
        if (d_means3d != nullptr) {
            cudaFree(d_means3d);
        }
        if (d_view != nullptr) {
            cudaFree(d_view);
        }
        if (d_proj != nullptr) {
            cudaFree(d_proj);
        }
        if (d_present != nullptr) {
            cudaFree(d_present);
        }
    };

    if (auto status = CopyHostToDevice(means3d.data(), means3d.size(), d_means3d); !status.ok()) {
        return status;
    }
    if (auto status = CopyHostToDevice(camera.pose.view_matrix.data(), camera.pose.view_matrix.size(), d_view);
        !status.ok()) {
        free_all();
        return status;
    }
    if (auto status = CopyHostToDevice(camera.pose.proj_matrix.data(), camera.pose.proj_matrix.size(), d_proj);
        !status.ok()) {
        free_all();
        return status;
    }

    const auto alloc_present_error = cudaMalloc(reinterpret_cast<void**>(&d_present), gaussians.size() * sizeof(bool));
    if (alloc_present_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("cudaMalloc failed for visibility output", alloc_present_error);
    }

    const auto memset_error = cudaMemset(d_present, 0, gaussians.size() * sizeof(bool));
    if (memset_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("cudaMemset failed for visibility output", memset_error);
    }

    CudaRasterizer::Rasterizer::markVisible(static_cast<int>(gaussians.size()), d_means3d, d_view, d_proj, d_present);

    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("markVisible kernel launch failed", launch_error);
    }

    const auto sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("markVisible kernel execution failed", sync_error);
    }

    std::unique_ptr<bool[]> host_present(new bool[gaussians.size()]);
    const auto copy_back_error =
        cudaMemcpy(host_present.get(), d_present, gaussians.size() * sizeof(bool), cudaMemcpyDeviceToHost);
    if (copy_back_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("cudaMemcpy device->host failed for visibility output", copy_back_error);
    }

    for (std::size_t i = 0; i < gaussians.size(); ++i) {
        present[i] = host_present[i];
    }

    free_all();
    return Status::Ok();
#endif
}

Status RasterizeForward(const RasterFrameInputs& inputs, RasterFrameOutputs& outputs) {
    outputs = {};

    if (inputs.gaussians == nullptr) {
        return Status::InvalidArgument("forward requires a non-null gaussian set");
    }
    if (inputs.camera == nullptr) {
        return Status::InvalidArgument("forward requires a non-null camera state");
    }

    const auto& gaussians = *inputs.gaussians;
    const auto& camera = *inputs.camera;
    const float scale_modifier = (inputs.gauss_config != nullptr) ? inputs.gauss_config->scale_modifier : 1.0f;

    if (const auto camera_status = ValidateCameraState(camera); !camera_status.ok()) {
        return camera_status;
    }

#if !defined(GS2PC_HAS_CUDA_RASTER)
    (void)gaussians;
    (void)camera;
    (void)scale_modifier;
    return Status::NotImplemented("CUDA rasterizer support is disabled in this build");
#else
    if (const auto device_status = EnsureCudaDevice(); !device_status.ok()) {
        return device_status;
    }

    const int P = static_cast<int>(gaussians.size());
    const int W = camera.render.image_width;
    const int H = camera.render.image_height;

    if (P == 0) {
        return Status::Ok();
    }

    std::vector<float> means3d;
    std::vector<float> colors;
    std::vector<float> opacities;
    std::vector<float> cov3d_precomp;

    means3d.reserve(static_cast<std::size_t>(P) * 3);
    colors.reserve(static_cast<std::size_t>(P) * 3);
    opacities.reserve(static_cast<std::size_t>(P));
    cov3d_precomp.reserve(static_cast<std::size_t>(P) * 6);

    const int sh_degree = std::clamp(inputs.sh_degree, 0, 4);
    const int max_sh_coeffs = 0;

    for (const auto& g : gaussians.items) {
        means3d.push_back(g.position.x);
        means3d.push_back(g.position.y);
        means3d.push_back(g.position.z);

        colors.push_back(g.color[0]);
        colors.push_back(g.color[1]);
        colors.push_back(g.color[2]);

        opacities.push_back(g.opacity);

        const float sx = std::exp(scale_modifier * g.scale.x);
        const float sy = std::exp(scale_modifier * g.scale.y);
        const float sz = std::exp(scale_modifier * g.scale.z);

        const float r = g.rotation[0];
        const float x = g.rotation[1];
        const float y = g.rotation[2];
        const float z = g.rotation[3];

        const float R00 = 1.0f - 2.0f * (y * y + z * z);
        const float R01 = 2.0f * (x * y - r * z);
        const float R02 = 2.0f * (x * z + r * y);
        const float R10 = 2.0f * (x * y + r * z);
        const float R11 = 1.0f - 2.0f * (x * x + z * z);
        const float R12 = 2.0f * (y * z - r * x);
        const float R20 = 2.0f * (x * z - r * y);
        const float R21 = 2.0f * (y * z + r * x);
        const float R22 = 1.0f - 2.0f * (x * x + y * y);

        const float L00 = R00 * sx;
        const float L01 = R01 * sy;
        const float L02 = R02 * sz;
        const float L10 = R10 * sx;
        const float L11 = R11 * sy;
        const float L12 = R12 * sz;
        const float L20 = R20 * sx;
        const float L21 = R21 * sy;
        const float L22 = R22 * sz;

        const float S00 = L00 * L00 + L01 * L01 + L02 * L02;
        const float S01 = L00 * L10 + L01 * L11 + L02 * L12;
        const float S02 = L00 * L20 + L01 * L21 + L02 * L22;
        const float S11 = L10 * L10 + L11 * L11 + L12 * L12;
        const float S12 = L10 * L20 + L11 * L21 + L12 * L22;
        const float S22 = L20 * L20 + L21 * L21 + L22 * L22;

        cov3d_precomp.push_back(S00);
        cov3d_precomp.push_back(S01);
        cov3d_precomp.push_back(S02);
        cov3d_precomp.push_back(S11);
        cov3d_precomp.push_back(S12);
        cov3d_precomp.push_back(S22);
    }

    std::vector<float> background = {1.0f, 1.0f, 1.0f};

    std::vector<int> mask;
    if (camera.mask.pixel_mask.size() == static_cast<std::size_t>(W * H)) {
        mask.assign(camera.mask.pixel_mask.begin(), camera.mask.pixel_mask.end());
    } else {
        mask.assign(static_cast<std::size_t>(W * H), 1);
    }

    float* d_background = nullptr;
    float* d_means3d = nullptr;
    float* d_colors = nullptr;
    float* d_opacities = nullptr;
    float* d_cov3d = nullptr;
    float* d_shs = nullptr;
    float* d_view = nullptr;
    float* d_proj = nullptr;
    float* d_campos = nullptr;
    int* d_mask = nullptr;

    float* d_out_color = nullptr;
    float* d_out_depth = nullptr;
    float* d_out_invdepth = nullptr;
    int* d_radii = nullptr;
    float* d_gauss_contrib = nullptr;
    float* d_gauss_surface = nullptr;
    int* d_gauss_pixels = nullptr;

    char* d_geom_buffer = nullptr;
    char* d_binning_buffer = nullptr;
    char* d_image_buffer = nullptr;
    size_t d_geom_buffer_size = 0;
    size_t d_binning_buffer_size = 0;
    size_t d_image_buffer_size = 0;

    const auto free_all = [&]() {
        if (d_background)
            cudaFree(d_background);
        if (d_means3d)
            cudaFree(d_means3d);
        if (d_colors)
            cudaFree(d_colors);
        if (d_opacities)
            cudaFree(d_opacities);
        if (d_cov3d)
            cudaFree(d_cov3d);
        if (d_shs)
            cudaFree(d_shs);
        if (d_view)
            cudaFree(d_view);
        if (d_proj)
            cudaFree(d_proj);
        if (d_campos)
            cudaFree(d_campos);
        if (d_mask)
            cudaFree(d_mask);
        if (d_out_color)
            cudaFree(d_out_color);
        if (d_out_depth)
            cudaFree(d_out_depth);
        if (d_out_invdepth)
            cudaFree(d_out_invdepth);
        if (d_radii)
            cudaFree(d_radii);
        if (d_gauss_contrib)
            cudaFree(d_gauss_contrib);
        if (d_gauss_surface)
            cudaFree(d_gauss_surface);
        if (d_gauss_pixels)
            cudaFree(d_gauss_pixels);
        if (d_geom_buffer)
            cudaFree(d_geom_buffer);
        if (d_binning_buffer)
            cudaFree(d_binning_buffer);
        if (d_image_buffer)
            cudaFree(d_image_buffer);
    };

    auto fail = [&](const Status& s) -> Status {
        free_all();
        return s;
    };

    if (auto s = CopyHostToDevice(background.data(), background.size(), d_background); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(means3d.data(), means3d.size(), d_means3d); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(colors.data(), colors.size(), d_colors); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(opacities.data(), opacities.size(), d_opacities); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(cov3d_precomp.data(), cov3d_precomp.size(), d_cov3d); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(camera.pose.view_matrix.data(), camera.pose.view_matrix.size(), d_view); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(camera.pose.proj_matrix.data(), camera.pose.proj_matrix.size(), d_proj); !s.ok())
        return fail(s);

    std::array<float, 3> campos = {camera.pose.camera_pos.x, camera.pose.camera_pos.y, camera.pose.camera_pos.z};
    if (auto s = CopyHostToDevice(campos.data(), campos.size(), d_campos); !s.ok())
        return fail(s);
    if (auto s = CopyHostToDevice(mask.data(), mask.size(), d_mask); !s.ok())
        return fail(s);

    const auto alloc_float = [&](float*& ptr, std::size_t n, const char* what) -> Status {
        const auto e = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float));
        return (e == cudaSuccess) ? Status::Ok() : MakeCudaErrorStatus(std::string("cudaMalloc failed for ") + what, e);
    };
    const auto alloc_int = [&](int*& ptr, std::size_t n, const char* what) -> Status {
        const auto e = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(int));
        return (e == cudaSuccess) ? Status::Ok() : MakeCudaErrorStatus(std::string("cudaMalloc failed for ") + what, e);
    };

    const std::size_t pixel_count = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
    if (auto s = alloc_float(d_out_color, pixel_count * 3, "out_color"); !s.ok())
        return fail(s);
    if (auto s = alloc_float(d_out_depth, pixel_count, "out_depth"); !s.ok())
        return fail(s);
    if (auto s = alloc_float(d_out_invdepth, pixel_count, "out_invdepth"); !s.ok())
        return fail(s);
    if (auto s = alloc_int(d_radii, static_cast<std::size_t>(P), "radii"); !s.ok())
        return fail(s);
    if (auto s = alloc_float(d_gauss_contrib, static_cast<std::size_t>(P), "gauss_contributions"); !s.ok())
        return fail(s);
    if (auto s = alloc_float(d_gauss_surface, static_cast<std::size_t>(P), "gauss_surface_distances"); !s.ok())
        return fail(s);
    if (auto s = alloc_int(d_gauss_pixels, static_cast<std::size_t>(P), "gauss_pixels"); !s.ok())
        return fail(s);

    cudaMemset(d_out_color, 0, pixel_count * 3 * sizeof(float));
    cudaMemset(d_out_depth, 0, pixel_count * sizeof(float));
    cudaMemset(d_out_invdepth, 0, pixel_count * sizeof(float));
    cudaMemset(d_radii, 0, static_cast<std::size_t>(P) * sizeof(int));
    cudaMemset(d_gauss_contrib, 0, static_cast<std::size_t>(P) * sizeof(float));

    std::vector<int> init_pixels(static_cast<std::size_t>(P), -1);
    const auto pixels_init_error =
        cudaMemcpy(d_gauss_pixels, init_pixels.data(), static_cast<std::size_t>(P) * sizeof(int), cudaMemcpyHostToDevice);
    if (pixels_init_error != cudaSuccess) {
        return fail(MakeCudaErrorStatus("cudaMemcpy failed for initial gauss pixel indices", pixels_init_error));
    }

    std::vector<float> init_surface(static_cast<std::size_t>(P), std::numeric_limits<float>::max());
    const auto surface_init_error = cudaMemcpy(d_gauss_surface, init_surface.data(),
                                               static_cast<std::size_t>(P) * sizeof(float), cudaMemcpyHostToDevice);
    if (surface_init_error != cudaSuccess) {
        return fail(MakeCudaErrorStatus("cudaMemcpy failed for initial gauss surface distances", surface_init_error));
    }

    std::function<char*(size_t)> geom_func = [&](size_t n) {
        if (n > d_geom_buffer_size) {
            if (d_geom_buffer != nullptr) {
                cudaFree(d_geom_buffer);
            }
            d_geom_buffer = nullptr;
            d_geom_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&d_geom_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            d_geom_buffer_size = n;
        }
        return d_geom_buffer;
    };
    std::function<char*(size_t)> binning_func = [&](size_t n) {
        if (n > d_binning_buffer_size) {
            if (d_binning_buffer != nullptr) {
                cudaFree(d_binning_buffer);
            }
            d_binning_buffer = nullptr;
            d_binning_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&d_binning_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            d_binning_buffer_size = n;
        }
        return d_binning_buffer;
    };
    std::function<char*(size_t)> image_func = [&](size_t n) {
        if (n > d_image_buffer_size) {
            if (d_image_buffer != nullptr) {
                cudaFree(d_image_buffer);
            }
            d_image_buffer = nullptr;
            d_image_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&d_image_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            d_image_buffer_size = n;
        }
        return d_image_buffer;
    };

    try {
        outputs.rendered = CudaRasterizer::Rasterizer::forward(
            geom_func, binning_func, image_func, P, sh_degree, max_sh_coeffs, d_background, W, H, d_means3d,
            nullptr, d_colors, d_opacities, nullptr, 1.0f, nullptr,
            d_cov3d, d_view, d_proj, d_campos, camera.intrinsics.tan_fov_x, camera.intrinsics.tan_fov_y,
            camera.render.prefiltered, d_out_color, d_out_depth, d_out_invdepth, camera.render.antialiasing,
            d_gauss_contrib, d_gauss_surface, d_gauss_pixels, d_mask, d_radii, inputs.calculate_surface_distance,
            camera.render.debug);
    } catch (const std::exception& e) {
        return fail(Status::RuntimeError(std::string("rasterizer forward threw: ") + e.what()));
    }

    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return fail(MakeCudaErrorStatus("forward kernel launch failed", launch_error));
    }

    const auto sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        return fail(MakeCudaErrorStatus("forward kernel execution failed", sync_error));
    }

    outputs.color.resize(pixel_count * 3);
    outputs.depth.resize(pixel_count);
    outputs.gauss_contributions.resize(static_cast<std::size_t>(P));
    outputs.gauss_surface_distances.resize(static_cast<std::size_t>(P));
    outputs.gauss_pixels.resize(static_cast<std::size_t>(P));

    auto copy_back = [&](void* dst, const void* src, std::size_t bytes, const char* what) -> Status {
        const auto e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return (e == cudaSuccess) ? Status::Ok()
                                  : MakeCudaErrorStatus(std::string("cudaMemcpy device->host failed for ") + what, e);
    };

    if (auto s = copy_back(outputs.color.data(), d_out_color, outputs.color.size() * sizeof(float), "out_color");
        !s.ok())
        return fail(s);
    if (auto s = copy_back(outputs.depth.data(), d_out_depth, outputs.depth.size() * sizeof(float), "out_depth");
        !s.ok())
        return fail(s);
    if (auto s = copy_back(outputs.gauss_contributions.data(), d_gauss_contrib,
                           outputs.gauss_contributions.size() * sizeof(float), "gauss_contributions");
        !s.ok())
        return fail(s);
    if (auto s = copy_back(outputs.gauss_surface_distances.data(), d_gauss_surface,
                           outputs.gauss_surface_distances.size() * sizeof(float), "gauss_surface_distances");
        !s.ok())
        return fail(s);
    if (auto s = copy_back(outputs.gauss_pixels.data(), d_gauss_pixels, outputs.gauss_pixels.size() * sizeof(int),
                           "gauss_pixels");
        !s.ok())
        return fail(s);

    free_all();
    return Status::Ok();
#endif
}

} // namespace gs2pc
