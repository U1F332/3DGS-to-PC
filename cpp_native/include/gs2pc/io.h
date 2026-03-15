#pragma once

#include "gs2pc/config.h"

#include <string>

namespace gs2pc {

[[nodiscard]] std::string BuildHelpText();
[[nodiscard]] Status ParseCommandLine(int argc, const char* const argv[], CliOptions& options);
[[nodiscard]] Status LoadGaussians(const std::filesystem::path& input_path, GaussianSet& gaussians);

} // namespace gs2pc
