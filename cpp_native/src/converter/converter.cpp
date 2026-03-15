#include "gs2pc/converter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace gs2pc {
namespace {

using Mat3 = std::array<std::array<float, 3>, 3>;
constexpr float kPi = 3.14159265358979323846f;

float ClampPositive(float v) {
    return v > 0.0f ? v : 0.0f;
}

Mat3 RotationFromQuaternion(const std::array<float, 4>& q) {
    const float r = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    return Mat3{{std::array<float, 3>{1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - r * z), 2.0f * (x * z + r * y)},
                 std::array<float, 3>{2.0f * (x * y + r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - r * x)},
                 std::array<float, 3>{2.0f * (x * z - r * y), 2.0f * (y * z + r * x), 1.0f - 2.0f * (x * x + y * y)}}};
}

Vec3f Multiply(const Mat3& m, const Vec3f& v) {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

Vec3f Normalize(const Vec3f& v) {
    const float n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (n <= std::numeric_limits<float>::epsilon()) {
        return {0.0f, 0.0f, 1.0f};
    }
    return {v.x / n, v.y / n, v.z / n};
}

Vec3f ComputeGaussianNormal(const Gaussian& g) {
    if (g.has_normal) {
        return Normalize(g.normal);
    }

    int min_axis = 0;
    if (g.scale.y < g.scale.x) {
        min_axis = 1;
    }
    if ((min_axis == 0 ? g.scale.x : g.scale.y) > g.scale.z) {
        min_axis = 2;
    }

    Vec3f axis{0.0f, 0.0f, 0.0f};
    if (min_axis == 0)
        axis.x = 1.0f;
    if (min_axis == 1)
        axis.y = 1.0f;
    if (min_axis == 2)
        axis.z = 1.0f;

    return Normalize(Multiply(RotationFromQuaternion(g.rotation), axis));
}

Mat3 BuildSamplingTransform(const Gaussian& g) {
    const Mat3 r = RotationFromQuaternion(g.rotation);
    const float sx = std::exp(g.scale.x);
    const float sy = std::exp(g.scale.y);
    const float sz = std::exp(g.scale.z);

    Mat3 l = r;
    l[0][0] *= sx;
    l[0][1] *= sy;
    l[0][2] *= sz;
    l[1][0] *= sx;
    l[1][1] *= sy;
    l[1][2] *= sz;
    l[2][0] *= sx;
    l[2][1] *= sy;
    l[2][2] *= sz;
    return l;
}

Vec3f SampleFromGaussian(const Gaussian& g, std::mt19937_64& rng) {
    static thread_local std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    const Mat3 l = BuildSamplingTransform(g);

    const Vec3f z = {normal_dist(rng), normal_dist(rng), normal_dist(rng)};
    const Vec3f delta = Multiply(l, z);

    return {g.position.x + delta.x, g.position.y + delta.y, g.position.z + delta.z};
}

float ZNorm(std::mt19937_64& rng, Vec3f& z) {
    static thread_local std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    z = {normal_dist(rng), normal_dist(rng), normal_dist(rng)};
    return std::sqrt(z.x * z.x + z.y * z.y + z.z * z.z);
}

float GaussianSurfaceWeight(const Gaussian& g, bool prioritise_visible) {
    const float sx = std::sqrt(std::max(std::exp(g.scale.x), 1e-8f));
    const float sy = std::sqrt(std::max(std::exp(g.scale.y), 1e-8f));
    const float sz = std::sqrt(std::max(std::exp(g.scale.z), 1e-8f));

    const float p = 1.6075f;
    const float ab = std::pow(sx * sy, p);
    const float ac = std::pow(sx * sz, p);
    const float bc = std::pow(sy * sz, p);
    const float radicand = std::max((ab + ac + bc) / 3.0f, 1e-12f);
    const float area = 4.0f * kPi * std::pow(radicand, 1.0f / p);

    const float contribution = prioritise_visible ? ClampPositive(g.opacity) : 1.0f;
    return std::sqrt(std::max(area, 0.0f)) * std::max(contribution, 1e-6f);
}

std::vector<std::size_t> BuildFilteredIndices(const GaussianSet& gaussians, const ConversionConfig& config) {
    std::vector<std::size_t> indices;
    indices.reserve(gaussians.size());

    for (std::size_t i = 0; i < gaussians.size(); ++i) {
        const auto& g = gaussians.items[i];

        if (g.opacity < config.min_opacity) {
            continue;
        }

        if (config.bounding_box_min.has_value()) {
            const auto& minv = *config.bounding_box_min;
            if (g.position.x < minv[0] || g.position.y < minv[1] || g.position.z < minv[2]) {
                continue;
            }
        }
        if (config.bounding_box_max.has_value()) {
            const auto& maxv = *config.bounding_box_max;
            if (g.position.x > maxv[0] || g.position.y > maxv[1] || g.position.z > maxv[2]) {
                continue;
            }
        }

        indices.push_back(i);
    }

    if (indices.empty()) {
        return indices;
    }

    if (config.cull_gaussian_sizes > 0.0f) {
        const float cull = std::clamp(config.cull_gaussian_sizes, 0.0f, 0.99f);
        const std::size_t keep_count =
            static_cast<std::size_t>(std::floor(static_cast<double>(indices.size()) * (1.0 - cull)));

        std::vector<std::pair<float, std::size_t>> weighted;
        weighted.reserve(indices.size());
        for (const auto idx : indices) {
            weighted.push_back({GaussianSurfaceWeight(gaussians.items[idx], config.prioritise_visible_gaussians), idx});
        }

        std::sort(weighted.begin(), weighted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<std::size_t> culled;
        culled.reserve(keep_count);
        for (std::size_t i = 0; i < keep_count && i < weighted.size(); ++i) {
            culled.push_back(weighted[i].second);
        }
        indices.swap(culled);
    }

    return indices;
}

std::vector<int> DistributePointCounts(const GaussianSet& gaussians, const std::vector<std::size_t>& indices,
                                       const ConversionConfig& config, int target_points) {
    std::vector<int> counts(indices.size(), 0);
    if (indices.empty() || target_points <= 0) {
        return counts;
    }

    std::vector<double> weights(indices.size(), 0.0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        weights[i] = static_cast<double>(
            GaussianSurfaceWeight(gaussians.items[indices[i]], config.prioritise_visible_gaussians));
    }

    const double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (wsum <= 0.0) {
        const int base = target_points / static_cast<int>(indices.size());
        int rem = target_points - base * static_cast<int>(indices.size());
        for (auto& c : counts) {
            c = base;
        }
        for (std::size_t i = 0; i < counts.size() && rem > 0; ++i, --rem) {
            counts[i] += 1;
        }
        return counts;
    }

    std::vector<double> raw(indices.size(), 0.0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        raw[i] = (weights[i] / wsum) * static_cast<double>(target_points);
    }

    if (config.exact_num_points) {
        std::vector<std::pair<double, std::size_t>> frac;
        frac.reserve(indices.size());
        int sum = 0;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            counts[i] = static_cast<int>(std::floor(raw[i]));
            sum += counts[i];
            frac.push_back({raw[i] - std::floor(raw[i]), i});
        }
        std::sort(frac.begin(), frac.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        int rem = target_points - sum;
        for (std::size_t i = 0; i < frac.size() && rem > 0; ++i, --rem) {
            counts[frac[i].second] += 1;
        }
        return counts;
    }

    int sum = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        counts[i] = static_cast<int>(std::round(raw[i]));
        sum += counts[i];
    }

    if (sum < target_points) {
        for (std::size_t i = 0; i < counts.size() && sum < target_points; ++i) {
            if (counts[i] == 0) {
                counts[i] = 1;
                ++sum;
            }
        }
    }

    return counts;
}

} // namespace

Status ConvertGaussiansToPointCloud(const GaussianSet& gaussians, const ConversionConfig& config,
                                    PointCloud& point_cloud, RenderStats* stats) {
    point_cloud.points.clear();

    const auto selected = BuildFilteredIndices(gaussians, config);
    if (selected.empty()) {
        if (stats != nullptr) {
            stats->visible_gaussian_count = 0;
            stats->contributed_gaussian_count = 0;
            stats->exported_point_count = 0;
        }
        return Status::Ok();
    }

    const int target_points = std::max(config.num_points, 1);
    const auto counts = DistributePointCounts(gaussians, selected, config, target_points);

    std::uint64_t planned_points = 0;
    for (const int c : counts) {
        if (c > 0) {
            planned_points += static_cast<std::uint64_t>(c);
        }
    }

    point_cloud.points.reserve(
        static_cast<std::size_t>(std::min<std::uint64_t>(planned_points, std::numeric_limits<std::size_t>::max())));

    std::mt19937_64 rng(0x20260312ULL);
    const int max_attempts = config.exact_num_points ? 100 : 5;

    std::size_t contributing_gaussians = 0;

    for (std::size_t i = 0; i < selected.size(); ++i) {
        const int count = counts[i];
        if (count <= 0) {
            continue;
        }

        const auto& g = gaussians.items[selected[i]];
        ++contributing_gaussians;

        const Vec3f normal = config.calculate_normals ? ComputeGaussianNormal(g) : Vec3f{};

        PointVertex center;
        center.position = g.position;
        center.color = g.color;
        center.normal = normal;
        center.has_normal = config.calculate_normals;
        point_cloud.points.push_back(center);

        const int remaining = count - 1;
        if (remaining <= 0) {
            continue;
        }

        const Mat3 l = BuildSamplingTransform(g);

        for (int s = 0; s < remaining; ++s) {
            bool accepted = false;
            PointVertex sampled{};
            sampled.color = g.color;
            sampled.normal = normal;
            sampled.has_normal = config.calculate_normals;

            for (int attempt = 0; attempt < max_attempts; ++attempt) {
                Vec3f z;
                const float norm = ZNorm(rng, z);
                if (norm > config.mahalanobis_distance_std) {
                    continue;
                }

                const Vec3f delta = Multiply(l, z);
                sampled.position = {g.position.x + delta.x, g.position.y + delta.y, g.position.z + delta.z};
                accepted = true;
                break;
            }

            if (accepted) {
                point_cloud.points.push_back(sampled);
            } else if (config.exact_num_points) {
                sampled.position = g.position;
                point_cloud.points.push_back(sampled);
            }
        }
    }

    if (stats != nullptr) {
        stats->visible_gaussian_count = static_cast<std::uint32_t>(selected.size());
        stats->contributed_gaussian_count = static_cast<std::uint32_t>(contributing_gaussians);
        stats->exported_point_count = static_cast<std::uint32_t>(point_cloud.points.size());
    }

    return Status::Ok();
}

} // namespace gs2pc
