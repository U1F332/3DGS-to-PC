#include "gs2pc/raster.h"

#include <algorithm>
#include <array>
#include <cmath>
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

template <typename T>
Status EnsureDeviceCapacity(T*& ptr, std::size_t& capacity, std::size_t required, const char* what) {
    if (required == 0) {
        return Status::Ok();
    }
    if (ptr != nullptr && capacity >= required) {
        return Status::Ok();
    }

    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
        capacity = 0;
    }

    const auto alloc_error = cudaMalloc(reinterpret_cast<void**>(&ptr), required * sizeof(T));
    if (alloc_error != cudaSuccess) {
        return MakeCudaErrorStatus(std::string("cudaMalloc failed for ") + what, alloc_error);
    }

    capacity = required;
    return Status::Ok();
}

struct CachedGaussianDeviceData {
    const GaussianSet* source = nullptr;
    int count = 0;
    float scale_modifier = 1.0f;

    float* d_means3d = nullptr;
    float* d_colors = nullptr;
    float* d_opacities = nullptr;
    float* d_cov3d = nullptr;

    std::size_t means_capacity = 0;
    std::size_t colors_capacity = 0;
    std::size_t opacities_capacity = 0;
    std::size_t cov_capacity = 0;

    void ResetIdentity() {
        source = nullptr;
        count = 0;
        scale_modifier = 1.0f;
    }

    void Release() {
        if (d_means3d) {
            cudaFree(d_means3d);
            d_means3d = nullptr;
        }
        if (d_colors) {
            cudaFree(d_colors);
            d_colors = nullptr;
        }
        if (d_opacities) {
            cudaFree(d_opacities);
            d_opacities = nullptr;
        }
        if (d_cov3d) {
            cudaFree(d_cov3d);
            d_cov3d = nullptr;
        }
        means_capacity = 0;
        colors_capacity = 0;
        opacities_capacity = 0;
        cov_capacity = 0;
        ResetIdentity();
    }

    ~CachedGaussianDeviceData() {
        Release();
    }
};

CachedGaussianDeviceData& GetGaussianCache() {
    static CachedGaussianDeviceData cache;
    return cache;
}

struct CachedForwardDeviceData {
    float* d_background = nullptr;
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
    float* d_best_contrib = nullptr;
    float* d_best_colors = nullptr;
    float* d_total_contrib = nullptr;
    float* d_min_surface = nullptr;

    char* d_geom_buffer = nullptr;
    char* d_binning_buffer = nullptr;
    char* d_image_buffer = nullptr;

    std::size_t background_capacity = 0;
    std::size_t view_capacity = 0;
    std::size_t proj_capacity = 0;
    std::size_t campos_capacity = 0;
    std::size_t mask_capacity = 0;

    std::size_t out_color_capacity = 0;
    std::size_t out_depth_capacity = 0;
    std::size_t out_invdepth_capacity = 0;
    std::size_t radii_capacity = 0;
    std::size_t gauss_contrib_capacity = 0;
    std::size_t gauss_surface_capacity = 0;
    std::size_t gauss_pixels_capacity = 0;
    std::size_t best_contrib_capacity = 0;
    std::size_t best_colors_capacity = 0;
    std::size_t total_contrib_capacity = 0;
    std::size_t min_surface_capacity = 0;

    std::size_t geom_buffer_size = 0;
    std::size_t binning_buffer_size = 0;
    std::size_t image_buffer_size = 0;

    const GaussianSet* accum_source = nullptr;
    int accum_count = 0;

    std::vector<float> init_surface_host;

    void Release() {
        if (d_background)
            cudaFree(d_background);
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
        if (d_best_contrib)
            cudaFree(d_best_contrib);
        if (d_best_colors)
            cudaFree(d_best_colors);
        if (d_total_contrib)
            cudaFree(d_total_contrib);
        if (d_min_surface)
            cudaFree(d_min_surface);

        d_background = nullptr;
        d_view = nullptr;
        d_proj = nullptr;
        d_campos = nullptr;
        d_mask = nullptr;

        d_out_color = nullptr;
        d_out_depth = nullptr;
        d_out_invdepth = nullptr;
        d_radii = nullptr;
        d_gauss_contrib = nullptr;
        d_gauss_surface = nullptr;
        d_gauss_pixels = nullptr;
        d_best_contrib = nullptr;
        d_best_colors = nullptr;
        d_total_contrib = nullptr;
        d_min_surface = nullptr;

        if (d_geom_buffer)
            cudaFree(d_geom_buffer);
        if (d_binning_buffer)
            cudaFree(d_binning_buffer);
        if (d_image_buffer)
            cudaFree(d_image_buffer);

        d_geom_buffer = nullptr;
        d_binning_buffer = nullptr;
        d_image_buffer = nullptr;

        background_capacity = 0;
        view_capacity = 0;
        proj_capacity = 0;
        campos_capacity = 0;
        mask_capacity = 0;

        out_color_capacity = 0;
        out_depth_capacity = 0;
        out_invdepth_capacity = 0;
        radii_capacity = 0;
        gauss_contrib_capacity = 0;
        gauss_surface_capacity = 0;
        gauss_pixels_capacity = 0;
        best_contrib_capacity = 0;
        best_colors_capacity = 0;
        total_contrib_capacity = 0;
        min_surface_capacity = 0;

        geom_buffer_size = 0;
        binning_buffer_size = 0;
        image_buffer_size = 0;

        accum_source = nullptr;
        accum_count = 0;
        init_surface_host.clear();
    }

    ~CachedForwardDeviceData() {
        Release();
    }
};

CachedForwardDeviceData& GetForwardCache() {
    static CachedForwardDeviceData cache;
    return cache;
}

Status EnsureForwardCacheCapacity(CachedForwardDeviceData& cache, int width, int height, int point_count) {
    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    if (auto s = EnsureDeviceCapacity(cache.d_background, cache.background_capacity, 3, "background"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_view, cache.view_capacity, 16, "view"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_proj, cache.proj_capacity, 16, "proj"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_campos, cache.campos_capacity, 3, "campos"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_mask, cache.mask_capacity, pixel_count, "mask"); !s.ok()) {
        cache.Release();
        return s;
    }

    if (auto s = EnsureDeviceCapacity(cache.d_out_color, cache.out_color_capacity, pixel_count * 3, "out_color"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_out_depth, cache.out_depth_capacity, pixel_count, "out_depth"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_out_invdepth, cache.out_invdepth_capacity, pixel_count, "out_invdepth");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_radii, cache.radii_capacity, static_cast<std::size_t>(point_count), "radii");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_contrib, cache.gauss_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_surface, cache.gauss_surface_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_surface_distances");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_pixels, cache.gauss_pixels_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_pixels");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_best_contrib, cache.best_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "best_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_best_colors, cache.best_colors_capacity,
                                      static_cast<std::size_t>(point_count) * 3, "best_colors");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_total_contrib, cache.total_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "total_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_min_surface, cache.min_surface_capacity,
                                      static_cast<std::size_t>(point_count), "min_surface_distances");
        !s.ok()) {
        cache.Release();
        return s;
    }

    if (cache.init_surface_host.size() < static_cast<std::size_t>(point_count)) {
        cache.init_surface_host.assign(static_cast<std::size_t>(point_count), std::numeric_limits<float>::max());
    }

    return Status::Ok();
}

Status ResetBestAccumulationIfNeeded(CachedForwardDeviceData& cache, const GaussianSet& gaussians, int point_count) {
    if (cache.accum_source == &gaussians && cache.accum_count == point_count) {
        return Status::Ok();
    }

    const auto reset_best_contrib =
        cudaMemset(cache.d_best_contrib, 0, static_cast<std::size_t>(point_count) * sizeof(float));
    if (reset_best_contrib != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for best contributions", reset_best_contrib);
    }

    const auto reset_best_colors =
        cudaMemset(cache.d_best_colors, 0, static_cast<std::size_t>(point_count) * 3 * sizeof(float));
    if (reset_best_colors != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for best colors", reset_best_colors);
    }

    const auto reset_total_contrib =
        cudaMemset(cache.d_total_contrib, 0, static_cast<std::size_t>(point_count) * sizeof(float));
    if (reset_total_contrib != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for total contributions", reset_total_contrib);
    }

    if (cache.init_surface_host.size() < static_cast<std::size_t>(point_count)) {
        cache.init_surface_host.assign(static_cast<std::size_t>(point_count), std::numeric_limits<float>::max());
    }

    const auto reset_min_surface = cudaMemcpy(cache.d_min_surface, cache.init_surface_host.data(),
                                              static_cast<std::size_t>(point_count) * sizeof(float),
                                              cudaMemcpyHostToDevice);
    if (reset_min_surface != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy failed for min surface reset", reset_min_surface);
    }

    cache.accum_source = &gaussians;
    cache.accum_count = point_count;
    return Status::Ok();
}

Status CopyCameraFrameToDevice(const CameraState& camera, int width, int height, CachedForwardDeviceData& cache) {
    const auto view_error =
        cudaMemcpy(cache.d_view, camera.pose.view_matrix.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (view_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for view", view_error);
    }

    const auto proj_error =
        cudaMemcpy(cache.d_proj, camera.pose.proj_matrix.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (proj_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for proj", proj_error);
    }

    const std::array<float, 3> campos = {camera.pose.camera_pos.x, camera.pose.camera_pos.y, camera.pose.camera_pos.z};
    const auto campos_error = cudaMemcpy(cache.d_campos, campos.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (campos_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for campos", campos_error);
    }

    std::vector<int> mask;
    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (camera.mask.pixel_mask.size() == pixel_count) {
        mask.assign(camera.mask.pixel_mask.begin(), camera.mask.pixel_mask.end());
    } else {
        mask.assign(pixel_count, 1);
    }

    const auto mask_error = cudaMemcpy(cache.d_mask, mask.data(), pixel_count * sizeof(int), cudaMemcpyHostToDevice);
    if (mask_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for mask", mask_error);
    }

    return Status::Ok();
}

Status InitializeForwardFrameBuffers(CachedForwardDeviceData& cache, int width, int height, int point_count) {
    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    const auto memset_color_error = cudaMemset(cache.d_out_color, 0, pixel_count * 3 * sizeof(float));
    if (memset_color_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for out_color", memset_color_error);
    }

    const auto memset_depth_error = cudaMemset(cache.d_out_depth, 0, pixel_count * sizeof(float));
    if (memset_depth_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for out_depth", memset_depth_error);
    }

    const auto memset_invdepth_error = cudaMemset(cache.d_out_invdepth, 0, pixel_count * sizeof(float));
    if (memset_invdepth_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for out_invdepth", memset_invdepth_error);
    }

    const auto memset_radii_error = cudaMemset(cache.d_radii, 0, static_cast<std::size_t>(point_count) * sizeof(int));
    if (memset_radii_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for radii", memset_radii_error);
    }

    const auto memset_contrib_error =
        cudaMemset(cache.d_gauss_contrib, 0, static_cast<std::size_t>(point_count) * sizeof(float));
    if (memset_contrib_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for gauss_contributions", memset_contrib_error);
    }

    const auto memset_pixels_error =
        cudaMemset(cache.d_gauss_pixels, 0xFF, static_cast<std::size_t>(point_count) * sizeof(int));
    if (memset_pixels_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemset failed for gauss_pixels", memset_pixels_error);
    }

    const auto surface_init_error = cudaMemcpy(cache.d_gauss_surface, cache.init_surface_host.data(),
                                               static_cast<std::size_t>(point_count) * sizeof(float), cudaMemcpyHostToDevice);
    if (surface_init_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy failed for initial gauss surface distances", surface_init_error);
    }

    return Status::Ok();
}

Status EnsureBackgroundOnDevice(CachedForwardDeviceData& cache) {
    const std::array<float, 3> background = {1.0f, 1.0f, 1.0f};
    const auto bg_error = cudaMemcpy(cache.d_background, background.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (bg_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for background", bg_error);
    }
    return Status::Ok();
}

Status UploadGaussiansIfNeeded(const GaussianSet& gaussians, float scale_modifier, CachedGaussianDeviceData& cache) {
    const int P = static_cast<int>(gaussians.size());
    if (P == 0) {
        cache.ResetIdentity();
        return Status::Ok();
    }

    const bool same_source = (cache.source == &gaussians);
    const bool same_count = (cache.count == P);
    const bool same_scale = (cache.scale_modifier == scale_modifier);

    if (same_source && same_count && same_scale && cache.d_means3d != nullptr && cache.d_colors != nullptr &&
        cache.d_opacities != nullptr && cache.d_cov3d != nullptr) {
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

    if (auto s = EnsureDeviceCapacity(cache.d_means3d, cache.means_capacity, means3d.size(), "means3d"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_colors, cache.colors_capacity, colors.size(), "colors"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_opacities, cache.opacities_capacity, opacities.size(), "opacities");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_cov3d, cache.cov_capacity, cov3d_precomp.size(), "cov3d"); !s.ok()) {
        cache.Release();
        return s;
    }

    const auto copy_means_error =
        cudaMemcpy(cache.d_means3d, means3d.data(), means3d.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (copy_means_error != cudaSuccess) {
        cache.Release();
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for means3d", copy_means_error);
    }

    const auto copy_colors_error =
        cudaMemcpy(cache.d_colors, colors.data(), colors.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (copy_colors_error != cudaSuccess) {
        cache.Release();
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for colors", copy_colors_error);
    }

    const auto copy_opacities_error =
        cudaMemcpy(cache.d_opacities, opacities.data(), opacities.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (copy_opacities_error != cudaSuccess) {
        cache.Release();
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for opacities", copy_opacities_error);
    }

    const auto copy_cov_error = cudaMemcpy(cache.d_cov3d, cov3d_precomp.data(), cov3d_precomp.size() * sizeof(float),
                                           cudaMemcpyHostToDevice);
    if (copy_cov_error != cudaSuccess) {
        cache.Release();
        return MakeCudaErrorStatus("cudaMemcpy host->device failed for cov3d", copy_cov_error);
    }

    cache.source = &gaussians;
    cache.count = P;
    cache.scale_modifier = scale_modifier;
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

    const float scale_modifier = (inputs.gauss_config != nullptr) ? inputs.gauss_config->scale_modifier : 1.0f;
    CachedGaussianDeviceData& gaussian_cache = GetGaussianCache();
    if (const auto upload_status = UploadGaussiansIfNeeded(gaussians, scale_modifier, gaussian_cache);
        !upload_status.ok()) {
        return upload_status;
    }

    float* d_view = nullptr;
    float* d_proj = nullptr;
    bool* d_present = nullptr;

    const auto free_all = [&]() {
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

    CudaRasterizer::Rasterizer::markVisible(static_cast<int>(gaussians.size()), gaussian_cache.d_means3d, d_view, d_proj,
                                            d_present);

    const auto launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        free_all();
        return MakeCudaErrorStatus("markVisible kernel launch failed", launch_error);
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

    CachedGaussianDeviceData& gaussian_cache = GetGaussianCache();
    if (const auto upload_status = UploadGaussiansIfNeeded(gaussians, scale_modifier, gaussian_cache);
        !upload_status.ok()) {
        return upload_status;
    }

    const int P = static_cast<int>(gaussians.size());
    const int W = camera.render.image_width;
    const int H = camera.render.image_height;

    if (P == 0) {
        return Status::Ok();
    }

    CachedForwardDeviceData& forward_cache = GetForwardCache();
    if (auto s = EnsureForwardCacheCapacity(forward_cache, W, H, P); !s.ok()) {
        return s;
    }
    if (auto s = ResetBestAccumulationIfNeeded(forward_cache, gaussians, P); !s.ok()) {
        return s;
    }
    if (auto s = EnsureBackgroundOnDevice(forward_cache); !s.ok()) {
        return s;
    }
    if (auto s = CopyCameraFrameToDevice(camera, W, H, forward_cache); !s.ok()) {
        return s;
    }
    if (auto s = InitializeForwardFrameBuffers(forward_cache, W, H, P); !s.ok()) {
        return s;
    }

    std::function<char*(size_t)> geom_func = [&](size_t n) {
        if (n > forward_cache.geom_buffer_size) {
            if (forward_cache.d_geom_buffer != nullptr) {
                cudaFree(forward_cache.d_geom_buffer);
            }
            forward_cache.d_geom_buffer = nullptr;
            forward_cache.geom_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&forward_cache.d_geom_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            forward_cache.geom_buffer_size = n;
        }
        return forward_cache.d_geom_buffer;
    };
    std::function<char*(size_t)> binning_func = [&](size_t n) {
        if (n > forward_cache.binning_buffer_size) {
            if (forward_cache.d_binning_buffer != nullptr) {
                cudaFree(forward_cache.d_binning_buffer);
            }
            forward_cache.d_binning_buffer = nullptr;
            forward_cache.binning_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&forward_cache.d_binning_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            forward_cache.binning_buffer_size = n;
        }
        return forward_cache.d_binning_buffer;
    };
    std::function<char*(size_t)> image_func = [&](size_t n) {
        if (n > forward_cache.image_buffer_size) {
            if (forward_cache.d_image_buffer != nullptr) {
                cudaFree(forward_cache.d_image_buffer);
            }
            forward_cache.d_image_buffer = nullptr;
            forward_cache.image_buffer_size = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&forward_cache.d_image_buffer), n) != cudaSuccess) {
                return static_cast<char*>(nullptr);
            }
            forward_cache.image_buffer_size = n;
        }
        return forward_cache.d_image_buffer;
    };

    try {
        outputs.rendered = CudaRasterizer::Rasterizer::forward(
            geom_func, binning_func, image_func, P, inputs.sh_degree, 0, forward_cache.d_background, W, H,
            gaussian_cache.d_means3d, nullptr, gaussian_cache.d_colors, gaussian_cache.d_opacities, nullptr, 1.0f,
            nullptr, gaussian_cache.d_cov3d, forward_cache.d_view, forward_cache.d_proj, forward_cache.d_campos,
            camera.intrinsics.tan_fov_x, camera.intrinsics.tan_fov_y, camera.render.prefiltered,
            forward_cache.d_out_color, forward_cache.d_out_depth, forward_cache.d_out_invdepth,
            camera.render.antialiasing, forward_cache.d_gauss_contrib, forward_cache.d_gauss_surface,
            forward_cache.d_gauss_pixels, forward_cache.d_mask, forward_cache.d_radii, inputs.calculate_surface_distance,
            camera.render.debug, forward_cache.d_best_contrib, forward_cache.d_best_colors,
            forward_cache.d_total_contrib, forward_cache.d_min_surface);
    } catch (const std::exception& e) {
        return Status::RuntimeError(std::string("rasterizer forward threw: ") + e.what());
    }

    outputs.color.clear();
    outputs.depth.clear();
    outputs.gauss_contributions.clear();
    outputs.gauss_surface_distances.clear();
    outputs.gauss_pixels.clear();

    return Status::Ok();
#endif
}

Status GetAccumulatedGaussianStatistics(std::size_t expected_count, bool include_surface,
                                       std::vector<float>& max_contributions,
                                       std::vector<float>& best_colours,
                                       std::vector<float>& total_contributions,
                                       std::vector<float>& min_surface_distances) {
#if !defined(GS2PC_HAS_CUDA_RASTER)
    (void)expected_count;
    (void)include_surface;
    max_contributions.clear();
    best_colours.clear();
    total_contributions.clear();
    min_surface_distances.clear();
    return Status::NotImplemented("CUDA rasterizer support is disabled in this build");
#else
    if (const auto device_status = EnsureCudaDevice(); !device_status.ok()) {
        return device_status;
    }

    CachedForwardDeviceData& forward_cache = GetForwardCache();
    if (forward_cache.d_best_contrib == nullptr || forward_cache.d_best_colors == nullptr ||
        forward_cache.d_total_contrib == nullptr || forward_cache.d_min_surface == nullptr) {
        max_contributions.clear();
        best_colours.clear();
        total_contributions.clear();
        min_surface_distances.clear();
        return Status::Ok();
    }

    if (forward_cache.accum_count <= 0) {
        max_contributions.clear();
        best_colours.clear();
        total_contributions.clear();
        min_surface_distances.clear();
        return Status::Ok();
    }

    const std::size_t count = static_cast<std::size_t>(forward_cache.accum_count);
    if (expected_count != 0 && expected_count != count) {
        return Status::InvalidArgument("accumulated gaussian count does not match expected_count");
    }

    max_contributions.resize(count);
    best_colours.resize(count * 3);
    total_contributions.resize(count);
    if (include_surface) {
        min_surface_distances.resize(count);
    } else {
        min_surface_distances.clear();
    }

    const auto max_error = cudaMemcpy(max_contributions.data(), forward_cache.d_best_contrib, count * sizeof(float),
                                      cudaMemcpyDeviceToHost);
    if (max_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy device->host failed for accumulated max contributions", max_error);
    }

    const auto col_error = cudaMemcpy(best_colours.data(), forward_cache.d_best_colors, count * 3 * sizeof(float),
                                      cudaMemcpyDeviceToHost);
    if (col_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy device->host failed for accumulated best colors", col_error);
    }

    const auto total_error = cudaMemcpy(total_contributions.data(), forward_cache.d_total_contrib,
                                        count * sizeof(float), cudaMemcpyDeviceToHost);
    if (total_error != cudaSuccess) {
        return MakeCudaErrorStatus("cudaMemcpy device->host failed for accumulated total contributions", total_error);
    }

    if (include_surface) {
        const auto min_surface_error = cudaMemcpy(min_surface_distances.data(), forward_cache.d_min_surface,
                                                  count * sizeof(float), cudaMemcpyDeviceToHost);
        if (min_surface_error != cudaSuccess) {
            return MakeCudaErrorStatus("cudaMemcpy device->host failed for accumulated min surface distances",
                                       min_surface_error);
        }
    }

    return Status::Ok();
#endif
}

Status EnsureForwardCacheCapacity(CachedForwardDeviceData& cache, int width, int height, int point_count) {
    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    if (auto s = EnsureDeviceCapacity(cache.d_background, cache.background_capacity, 3, "background"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_view, cache.view_capacity, 16, "view"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_proj, cache.proj_capacity, 16, "proj"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_campos, cache.campos_capacity, 3, "campos"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_mask, cache.mask_capacity, pixel_count, "mask"); !s.ok()) {
        cache.Release();
        return s;
    }

    if (auto s = EnsureDeviceCapacity(cache.d_out_color, cache.out_color_capacity, pixel_count * 3, "out_color"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_out_depth, cache.out_depth_capacity, pixel_count, "out_depth"); !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_out_invdepth, cache.out_invdepth_capacity, pixel_count, "out_invdepth");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_radii, cache.radii_capacity, static_cast<std::size_t>(point_count), "radii");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_contrib, cache.gauss_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_surface, cache.gauss_surface_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_surface_distances");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_gauss_pixels, cache.gauss_pixels_capacity,
                                      static_cast<std::size_t>(point_count), "gauss_pixels");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_best_contrib, cache.best_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "best_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_best_colors, cache.best_colors_capacity,
                                      static_cast<std::size_t>(point_count) * 3, "best_colors");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_total_contrib, cache.total_contrib_capacity,
                                      static_cast<std::size_t>(point_count), "total_contributions");
        !s.ok()) {
        cache.Release();
        return s;
    }
    if (auto s = EnsureDeviceCapacity(cache.d_min_surface, cache.min_surface_capacity,
                                      static_cast<std::size_t>(point_count), "min_surface_distances");
        !s.ok()) {
        cache.Release();
        return s;
    }

    if (cache.init_surface_host.size() < static_cast<std::size_t>(point_count)) {
        cache.init_surface_host.assign(static_cast<std::size_t>(point_count), std::numeric_limits<float>::max());
    }

    return Status::Ok();
}

} // namespace gs2pc

