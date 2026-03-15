#include "gs2pc/io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

namespace gs2pc {
namespace {

constexpr float kShC0 = 0.28209479177387814f;

float Clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float Sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

void NormalizeQuaternion(std::array<float, 4>& q) {
    const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm <= std::numeric_limits<float>::epsilon()) {
        q = {1.0f, 0.0f, 0.0f, 0.0f};
        return;
    }

    for (float& component : q) {
        component /= norm;
    }
}

std::array<float, 3> ComputeColourFromDc(float c0, float c1, float c2) {
    return {Clamp01(kShC0 * c0 + 0.5f), Clamp01(kShC0 * c1 + 0.5f), Clamp01(kShC0 * c2 + 0.5f)};
}

bool AllTrue(const std::array<bool, 3>& values) {
    return values[0] && values[1] && values[2];
}

bool AllTrue(const std::array<bool, 4>& values) {
    return values[0] && values[1] && values[2] && values[3];
}

struct SplatRecord {
    float xyz[3];
    float scales[3];
    std::uint8_t colour[4];
    std::uint8_t rots[4];
};

Status LoadSplat(const std::filesystem::path& input_path, GaussianSet& gaussians) {
    std::ifstream stream(input_path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return Status::IoError("failed to open input file: " + input_path.string());
    }

    const auto size = stream.tellg();
    if (size < 0) {
        return Status::IoError("failed to determine file size: " + input_path.string());
    }

    if ((static_cast<std::uint64_t>(size) % sizeof(SplatRecord)) != 0) {
        return Status::IoError("invalid .splat file size: " + input_path.string());
    }

    stream.seekg(0, std::ios::beg);
    const std::size_t record_count = static_cast<std::size_t>(size) / sizeof(SplatRecord);
    std::vector<SplatRecord> records(record_count);
    if (!records.empty()) {
        stream.read(reinterpret_cast<char*>(records.data()),
                    static_cast<std::streamsize>(records.size() * sizeof(SplatRecord)));
    }

    if (!stream && !records.empty()) {
        return Status::IoError("failed to read .splat data: " + input_path.string());
    }

    gaussians = {};
    gaussians.source_path = input_path;
    gaussians.source_kind = InputFileKind::kSplat;
    gaussians.has_gaussian_attributes = true;
    gaussians.items.reserve(record_count);

    for (const auto& record : records) {
        Gaussian gaussian;
        gaussian.position = {record.xyz[0], record.xyz[1], record.xyz[2]};
        gaussian.scale = {std::log(std::max(record.scales[0], std::numeric_limits<float>::min())),
                          std::log(std::max(record.scales[1], std::numeric_limits<float>::min())),
                          std::log(std::max(record.scales[2], std::numeric_limits<float>::min()))};
        gaussian.color = {record.colour[0] / 255.0f, record.colour[1] / 255.0f, record.colour[2] / 255.0f};
        gaussian.opacity = record.colour[3] / 255.0f;
        gaussian.rotation = {(static_cast<float>(record.rots[0]) - 128.0f) / 128.0f,
                             (static_cast<float>(record.rots[1]) - 128.0f) / 128.0f,
                             (static_cast<float>(record.rots[2]) - 128.0f) / 128.0f,
                             (static_cast<float>(record.rots[3]) - 128.0f) / 128.0f};
        NormalizeQuaternion(gaussian.rotation);
        gaussians.items.push_back(std::move(gaussian));
    }

    return Status::Ok();
}

enum class PlyFormat { kAscii, kBinaryLittleEndian };

enum class PlyScalarType { kInt8, kUInt8, kInt16, kUInt16, kInt32, kUInt32, kFloat32, kFloat64 };

struct PlyProperty {
    std::string name;
    PlyScalarType type;
};

struct PlyHeader {
    PlyFormat format = PlyFormat::kBinaryLittleEndian;
    std::size_t vertex_count = 0;
    std::vector<PlyProperty> vertex_properties;
    std::streampos data_start = 0;
    std::array<bool, 3> has_normals = {false, false, false};
    std::array<bool, 3> has_rgb = {false, false, false};
    std::array<bool, 3> has_scale = {false, false, false};
    std::array<bool, 4> has_rotation = {false, false, false, false};
    bool has_opacity = false;
    int dc_count = 0;
    int rest_count = 0;
    InputFileKind input_kind = InputFileKind::kUnknown;
};

std::string TrimLine(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

InputFileKind DetectPlyInputKind(const PlyHeader& header) {
    const bool gaussian_core = header.has_opacity && AllTrue(header.has_scale) && AllTrue(header.has_rotation);
    return gaussian_core ? InputFileKind::kGaussianPly : InputFileKind::kPointCloudPly;
}

Status ParsePlyScalarType(std::string_view text, PlyScalarType& type) {
    if (text == "char" || text == "int8") {
        type = PlyScalarType::kInt8;
        return Status::Ok();
    }
    if (text == "uchar" || text == "uint8") {
        type = PlyScalarType::kUInt8;
        return Status::Ok();
    }
    if (text == "short" || text == "int16") {
        type = PlyScalarType::kInt16;
        return Status::Ok();
    }
    if (text == "ushort" || text == "uint16") {
        type = PlyScalarType::kUInt16;
        return Status::Ok();
    }
    if (text == "int" || text == "int32") {
        type = PlyScalarType::kInt32;
        return Status::Ok();
    }
    if (text == "uint" || text == "uint32") {
        type = PlyScalarType::kUInt32;
        return Status::Ok();
    }
    if (text == "float" || text == "float32") {
        type = PlyScalarType::kFloat32;
        return Status::Ok();
    }
    if (text == "double" || text == "float64") {
        type = PlyScalarType::kFloat64;
        return Status::Ok();
    }
    return Status::InvalidArgument("unsupported PLY scalar type: " + std::string(text));
}

Status ParsePlyHeader(std::ifstream& stream, PlyHeader& header) {
    std::string line;
    if (!std::getline(stream, line)) {
        return Status::IoError("failed to read PLY header");
    }
    if (TrimLine(line) != "ply") {
        return Status::InvalidArgument("input file is not a valid PLY file");
    }

    bool in_vertex_element = false;
    bool saw_format = false;
    bool saw_vertex = false;

    while (std::getline(stream, line)) {
        line = TrimLine(std::move(line));
        if (line.empty()) {
            continue;
        }
        if (line == "end_header") {
            header.data_start = stream.tellg();
            if (!saw_format) {
                return Status::InvalidArgument("PLY header missing format declaration");
            }
            if (!saw_vertex) {
                return Status::InvalidArgument("PLY header missing vertex element");
            }
            header.input_kind = DetectPlyInputKind(header);
            return Status::Ok();
        }

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "comment" || keyword == "obj_info") {
            continue;
        }
        if (keyword == "format") {
            std::string format_name;
            std::string version;
            iss >> format_name >> version;
            if (format_name == "ascii") {
                header.format = PlyFormat::kAscii;
            } else if (format_name == "binary_little_endian") {
                header.format = PlyFormat::kBinaryLittleEndian;
            } else {
                return Status::InvalidArgument("unsupported PLY format: " + format_name);
            }
            saw_format = true;
            continue;
        }
        if (keyword == "element") {
            std::string element_name;
            std::size_t count = 0;
            iss >> element_name >> count;
            in_vertex_element = (element_name == "vertex");
            if (in_vertex_element) {
                header.vertex_count = count;
                saw_vertex = true;
            }
            continue;
        }
        if (keyword == "property") {
            if (!in_vertex_element) {
                continue;
            }

            std::string type_name;
            iss >> type_name;
            if (type_name == "list") {
                return Status::InvalidArgument("list properties inside vertex element are not supported");
            }

            std::string property_name;
            iss >> property_name;
            PlyScalarType type = PlyScalarType::kFloat32;
            if (const auto status = ParsePlyScalarType(type_name, type); !status.ok()) {
                return status;
            }
            header.vertex_properties.push_back({property_name, type});

            if (property_name == "nx") {
                header.has_normals[0] = true;
            } else if (property_name == "ny") {
                header.has_normals[1] = true;
            } else if (property_name == "nz") {
                header.has_normals[2] = true;
            } else if (property_name == "red") {
                header.has_rgb[0] = true;
            } else if (property_name == "green") {
                header.has_rgb[1] = true;
            } else if (property_name == "blue") {
                header.has_rgb[2] = true;
            } else if (property_name == "opacity") {
                header.has_opacity = true;
            } else if (property_name == "scale_0") {
                header.has_scale[0] = true;
            } else if (property_name == "scale_1") {
                header.has_scale[1] = true;
            } else if (property_name == "scale_2") {
                header.has_scale[2] = true;
            } else if (property_name == "rot_0") {
                header.has_rotation[0] = true;
            } else if (property_name == "rot_1") {
                header.has_rotation[1] = true;
            } else if (property_name == "rot_2") {
                header.has_rotation[2] = true;
            } else if (property_name == "rot_3") {
                header.has_rotation[3] = true;
            } else if (property_name.rfind("f_dc_", 0) == 0) {
                header.dc_count = std::max(header.dc_count, std::stoi(property_name.substr(5)) + 1);
            } else if (property_name.rfind("f_rest_", 0) == 0) {
                header.rest_count = std::max(header.rest_count, std::stoi(property_name.substr(7)) + 1);
            }
            continue;
        }
    }

    return Status::InvalidArgument("PLY header is missing end_header");
}

Status ReadBinaryScalar(std::ifstream& stream, PlyScalarType type, float& value) {
    switch (type) {
    case PlyScalarType::kInt8: {
        std::int8_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kUInt8: {
        std::uint8_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kInt16: {
        std::int16_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kUInt16: {
        std::uint16_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kInt32: {
        std::int32_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kUInt32: {
        std::uint32_t v = 0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    case PlyScalarType::kFloat32: {
        float v = 0.0f;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = v;
        break;
    }
    case PlyScalarType::kFloat64: {
        double v = 0.0;
        stream.read(reinterpret_cast<char*>(&v), sizeof(v));
        value = static_cast<float>(v);
        break;
    }
    }

    if (!stream) {
        return Status::IoError("failed while reading binary PLY vertex data");
    }

    return Status::Ok();
}

Status ParseAsciiScalar(std::string_view token, float& value) {
    try {
        std::size_t processed = 0;
        value = std::stof(std::string(token), &processed);
        if (processed != token.size()) {
            return Status::InvalidArgument("invalid ASCII PLY numeric token");
        }
    } catch (...) {
        return Status::InvalidArgument("invalid ASCII PLY numeric token");
    }
    return Status::Ok();
}

void AssignPlyProperty(const PlyProperty& property, float value, Gaussian& gaussian, std::array<bool, 3>& has_rgb,
                       std::array<float, 3>& rgb, std::array<bool, 3>& has_dc, std::array<float, 3>& dc,
                       std::array<bool, 3>& has_normal) {
    const std::string_view name = property.name;
    if (name == "x") {
        gaussian.position.x = value;
        return;
    }
    if (name == "y") {
        gaussian.position.y = value;
        return;
    }
    if (name == "z") {
        gaussian.position.z = value;
        return;
    }
    if (name == "nx") {
        gaussian.normal.x = value;
        has_normal[0] = true;
        return;
    }
    if (name == "ny") {
        gaussian.normal.y = value;
        has_normal[1] = true;
        return;
    }
    if (name == "nz") {
        gaussian.normal.z = value;
        has_normal[2] = true;
        return;
    }
    if (name == "opacity") {
        gaussian.opacity = Sigmoid(value);
        return;
    }
    if (name == "scale_0") {
        gaussian.scale.x = value;
        return;
    }
    if (name == "scale_1") {
        gaussian.scale.y = value;
        return;
    }
    if (name == "scale_2") {
        gaussian.scale.z = value;
        return;
    }
    if (name == "rot_0") {
        gaussian.rotation[0] = value;
        return;
    }
    if (name == "rot_1") {
        gaussian.rotation[1] = value;
        return;
    }
    if (name == "rot_2") {
        gaussian.rotation[2] = value;
        return;
    }
    if (name == "rot_3") {
        gaussian.rotation[3] = value;
        return;
    }
    if (name == "red") {
        rgb[0] = value;
        has_rgb[0] = true;
        return;
    }
    if (name == "green") {
        rgb[1] = value;
        has_rgb[1] = true;
        return;
    }
    if (name == "blue") {
        rgb[2] = value;
        has_rgb[2] = true;
        return;
    }
    if (name == "f_dc_0") {
        dc[0] = value;
        has_dc[0] = true;
        gaussian.sh_coefficients.push_back(value);
        return;
    }
    if (name == "f_dc_1") {
        dc[1] = value;
        has_dc[1] = true;
        gaussian.sh_coefficients.push_back(value);
        return;
    }
    if (name == "f_dc_2") {
        dc[2] = value;
        has_dc[2] = true;
        gaussian.sh_coefficients.push_back(value);
        return;
    }
    if (name.starts_with("f_rest_")) {
        gaussian.sh_coefficients.push_back(value);
    }
}

void FinalizePlyGaussian(Gaussian& gaussian, const std::array<bool, 3>& has_rgb, const std::array<float, 3>& rgb,
                         const std::array<bool, 3>& has_dc, const std::array<float, 3>& dc,
                         const std::array<bool, 3>& has_normal) {
    if (has_rgb[0] && has_rgb[1] && has_rgb[2]) {
        const bool needs_normalization = rgb[0] > 1.0f || rgb[1] > 1.0f || rgb[2] > 1.0f;
        const float scale = needs_normalization ? (1.0f / 255.0f) : 1.0f;
        gaussian.color = {Clamp01(rgb[0] * scale), Clamp01(rgb[1] * scale), Clamp01(rgb[2] * scale)};
    } else if (has_dc[0] && has_dc[1] && has_dc[2]) {
        gaussian.color = ComputeColourFromDc(dc[0], dc[1], dc[2]);
    }

    gaussian.has_normal = has_normal[0] && has_normal[1] && has_normal[2];
    NormalizeQuaternion(gaussian.rotation);
}

Status LoadPly(const std::filesystem::path& input_path, GaussianSet& gaussians) {
    std::ifstream stream(input_path, std::ios::binary);
    if (!stream) {
        return Status::IoError("failed to open input file: " + input_path.string());
    }

    PlyHeader header;
    if (const auto status = ParsePlyHeader(stream, header); !status.ok()) {
        return status;
    }

    gaussians = {};
    gaussians.source_path = input_path;
    gaussians.source_kind = header.input_kind;
    gaussians.has_normals = AllTrue(header.has_normals);
    gaussians.has_sh_coefficients = (header.dc_count > 0 || header.rest_count > 0);
    gaussians.has_gaussian_attributes = (header.input_kind == InputFileKind::kGaussianPly);
    gaussians.items.reserve(header.vertex_count);

    if (header.format == PlyFormat::kBinaryLittleEndian) {
        stream.seekg(header.data_start);
        for (std::size_t i = 0; i < header.vertex_count; ++i) {
            Gaussian gaussian;
            std::array<bool, 3> has_rgb = {false, false, false};
            std::array<float, 3> rgb = {0.0f, 0.0f, 0.0f};
            std::array<bool, 3> has_dc = {false, false, false};
            std::array<float, 3> dc = {0.0f, 0.0f, 0.0f};
            std::array<bool, 3> has_normal = {false, false, false};

            for (const auto& property : header.vertex_properties) {
                float value = 0.0f;
                if (const auto status = ReadBinaryScalar(stream, property.type, value); !status.ok()) {
                    return status;
                }
                AssignPlyProperty(property, value, gaussian, has_rgb, rgb, has_dc, dc, has_normal);
            }

            FinalizePlyGaussian(gaussian, has_rgb, rgb, has_dc, dc, has_normal);
            gaussians.items.push_back(std::move(gaussian));
        }
        return Status::Ok();
    }

    stream.clear();
    stream.seekg(header.data_start);
    std::string line;
    for (std::size_t i = 0; i < header.vertex_count; ++i) {
        if (!std::getline(stream, line)) {
            return Status::IoError("unexpected end of ASCII PLY vertex data");
        }
        line = TrimLine(std::move(line));
        if (line.empty()) {
            --i;
            continue;
        }

        std::istringstream iss(line);
        Gaussian gaussian;
        std::array<bool, 3> has_rgb = {false, false, false};
        std::array<float, 3> rgb = {0.0f, 0.0f, 0.0f};
        std::array<bool, 3> has_dc = {false, false, false};
        std::array<float, 3> dc = {0.0f, 0.0f, 0.0f};
        std::array<bool, 3> has_normal = {false, false, false};

        for (const auto& property : header.vertex_properties) {
            std::string token;
            if (!(iss >> token)) {
                return Status::InvalidArgument("ASCII PLY vertex row has fewer columns than expected");
            }
            float value = 0.0f;
            if (const auto status = ParseAsciiScalar(token, value); !status.ok()) {
                return status;
            }
            AssignPlyProperty(property, value, gaussian, has_rgb, rgb, has_dc, dc, has_normal);
        }

        FinalizePlyGaussian(gaussian, has_rgb, rgb, has_dc, dc, has_normal);
        gaussians.items.push_back(std::move(gaussian));
    }

    return Status::Ok();
}

} // namespace

Status LoadGaussians(const std::filesystem::path& input_path, GaussianSet& gaussians) {
    if (input_path.empty()) {
        return Status::InvalidArgument("input path is empty");
    }
    if (!std::filesystem::exists(input_path)) {
        return Status::IoError("input file does not exist: " + input_path.string());
    }

    const auto extension = input_path.extension().string();
    if (extension == ".splat") {
        return LoadSplat(input_path, gaussians);
    }
    if (extension == ".ply") {
        return LoadPly(input_path, gaussians);
    }

    return Status::InvalidArgument("unsupported input file type: " + extension);
}

} // namespace gs2pc
