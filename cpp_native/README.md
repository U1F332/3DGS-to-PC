# `cpp_native` detailed notes

## Overview

`cpp_native/` is the native C++20 implementation scaffold for this repository.

It provides:
- a reusable library target: `gs2pc_core`
- a command-line executable: `gs2pc_cli`
- native loaders for Gaussian `.ply` and `.splat` inputs
- COLMAP camera loading for text and binary formats
- point-cloud generation and PLY export
- optional CUDA raster diagnostic integration

This directory is built with CMake and is intended to be configured with the Ninja generator in this workspace.

## Functional consistency with the original implementation

The original implementation in this repository is the Python pipeline centered around `gauss_to_pc.py`.

The current `cpp_native/` implementation now reproduces the main Gaussian-to-point-cloud path much more closely than the earlier native scaffold, especially for colour extraction when CUDA raster support is available.

### Areas that are already broadly aligned
- Both paths target conversion from Gaussian scene data to exported point clouds.
- Both paths accept Gaussian `.ply` and `.splat` style inputs.
- Both paths include support for COLMAP-derived camera data.
- Both paths expose controls related to point count, opacity filtering, bounding box filtering, SH-related settings, and visibility-oriented options.
- Both paths can write point-cloud PLY output.
- Both paths can now use camera-driven colour extraction rather than relying only on static stored Gaussian colour.

### Current colour extraction behavior in `cpp_native/`
- When CUDA raster support is available and a transform path is provided, `gs2pc_cli` now automatically uses the CUDA forward raster path as the main colour extraction path.
- In that mode, the native path:
  - loads COLMAP camera frames
  - passes Gaussian SH coefficients into the CUDA rasterizer when available
  - renders across all available camera frames
  - tracks per-Gaussian maximum contribution values
  - records the best contributing pixel for each Gaussian
  - writes the selected rendered colour back to the Gaussian before point-cloud export
- This is much closer to the original repository behavior than the earlier native SH-view averaging fallback.
- If CUDA raster support is unavailable, the native path falls back to a camera-aware SH colour averaging path instead of failing silently.

### Areas where `cpp_native/` now has strong practical alignment
- Native loading and parsing of Gaussian PLY and SPLAT data are in good shape.
- COLMAP camera loading is implemented for both text and binary formats.
- Point sampling and point-cloud export are operational and stable.
- CUDA forward output parsing is integrated into the native main flow rather than being limited to a standalone diagnostic-only path.
- Render-driven colour extraction is now part of the native main workflow when CUDA is enabled.

### Areas where differences may still remain
- The Python path still remains the fuller reference implementation for the entire repository feature set.
- Mask loading and mask-driven colour extraction are not yet fully mirrored in the native path.
- Mesh generation and point-cloud post-processing utilities from the Python workflow are not yet fully reproduced in native code.
- Some edge-case differences may still exist in visibility filtering, surface-distance handling, threshold semantics, or floating-point accumulation details.
- Exact equivalence should still be treated as a result to verify empirically on the same dataset rather than assumed purely from implementation intent.

### Practical assessment
- Input compatibility: good
- CLI surface compatibility: good
- Native loading and export coverage: good
- Rendering parity with the Python pipeline: high when CUDA forward is enabled
- Mesh and post-processing parity: not yet achieved
- Overall Gaussian-to-point-cloud parity for the main conversion path: high

### Conclusion

`cpp_native/` is no longer only a minimal native scaffold for the core conversion path.

For the main use case of converting Gaussians into a dense point cloud with camera-driven colour extraction, the native implementation is now much closer to the original repository behavior, especially when built with CUDA raster support enabled.

The Python implementation still remains the broader reference for auxiliary features such as masking, meshing, and some higher-level processing details, but the core native conversion path is now substantially aligned.

## Build directory convention

The build directory for this native component should be named `build`.

Recommended configure command:

- `cmake -S . -B build -G Ninja`

Recommended build command:

- `cmake --build build`

In this repository, using a directory such as `build_ninja` is unnecessary because the generator is already explicitly selected by `-G Ninja`.

## Why the IDE solution-wide build command is not the right path here

This native component is a standalone CMake project rooted at `cpp_native/CMakeLists.txt`.

That means the expected build flow is:
- configure with CMake
- generate Ninja files
- build through CMake or Ninja

So the correct command path is based on CMake, for example:
- `cmake -S C:/WS/Test/3DGS-to-PC/cpp_native -B C:/WS/Test/3DGS-to-PC/cpp_native/build -G Ninja`
- `cmake --build C:/WS/Test/3DGS-to-PC/cpp_native/build`

## Toolchain and language settings

Current native project settings:
- minimum CMake version: `3.8`
- configured CMake range in `cpp_native/CMakeLists.txt`: `3.8...3.31`
- language standard: `C++20`
- generator used in this workspace: `Ninja`

## Current source layout

### Public headers
- `include/gs2pc/config.h`
- `include/gs2pc/types.h`
- `include/gs2pc/io.h`
- `include/gs2pc/camera.h`
- `include/gs2pc/converter.h`
- `include/gs2pc/exporter.h`
- `include/gs2pc/raster.h`

### Source files
- `src/cli/cli.cpp`
- `src/cli/main.cpp`
- `src/io/io.cpp`
- `src/camera/camera_loader.cpp`
- `src/converter/converter.cpp`
- `src/exporter/exporter.cpp`
- `src/raster/raster.cpp`

## What each part currently does

### `src/io/io.cpp`
Responsible for reading Gaussian input data.

Current implemented support:
- binary `.splat`
- ASCII PLY
- binary little-endian PLY

It detects whether a PLY file is:
- a point-cloud PLY
- a Gaussian-style PLY

The loader reads practical attributes when present, including:
- position
- normals
- RGB colour
- opacity
- Gaussian scales
- quaternion rotation
- SH DC terms and additional SH coefficients

### `src/camera/camera_loader.cpp`
Responsible for camera loading.

Current implemented support:
- `cameras.txt`
- `images.txt`
- `cameras.bin`
- `images.bin`

This is intended for COLMAP camera and image metadata loading.

### `src/converter/converter.cpp`
Responsible for converting loaded Gaussians into a point cloud.

Current implemented behavior includes:
- filtering by minimum opacity
- filtering by bounding box
- culling a percentage of the largest Gaussians
- distributing target point counts across selected Gaussians
- exact and non-exact point count distribution modes
- Gaussian sampling using rotation and scale
- normal generation or reuse when available

### `src/exporter/exporter.cpp`
Responsible for writing point clouds as PLY.

Current output behavior:
- always writes vertex positions
- writes normals when available
- writes RGB colour as `uchar`
- supports both binary little-endian and ASCII PLY output; binary is the default
- accepts an optional `ExportTimings*` pointer to receive per-stage write timings (header, vertices, flush, total)

### `src/cli/cli.cpp`
Responsible for command-line parsing and help text.

The CLI currently exposes options for:
- input and output paths
- transform and mask paths
- target point count
- visibility and surface-distance thresholds
- opacity and size filtering
- SH degree
- colour quality presets
- image dimensions and diagnostic camera settings
- enabling mark-visible and forward diagnostic paths
- PLY export format (`--export_format <binary|ascii>`, default: `binary`)

### `src/raster/raster.cpp`
Responsible for optional native CUDA raster integration.

Important behavior:
- when CUDA raster support is disabled, diagnostic functions return a not-implemented status
- when enabled, the code can call native `markVisible` and forward raster paths
- CUDA support is controlled through `GS2PC_ENABLE_CUDA_RASTER`

## Build targets

Defined targets in `cpp_native/CMakeLists.txt`:
- `gs2pc_core`
- `gs2pc_cli`

Optional formatting target:
- `gs2pc_format`

The formatting target is only created when `clang-format` is found.

## Optional CUDA support

CUDA raster support is disabled by default.

Relevant CMake options:
- `GS2PC_ENABLE_CUDA_RASTER=ON|OFF`
- `GS2PC_CUDA_MIN_SM`
- `GS2PC_FORCE_MIN_CUDA_ARCH=ON|OFF`

Important behavior:
- when CUDA raster support is enabled successfully, `gs2pc_cli` can use CUDA forward rendering as the main colour extraction path when camera transforms are provided
- the native raster path now supports passing SH coefficients into the CUDA rasterizer instead of relying only on precomputed colours
- if CUDA is unavailable, the project still builds, but the native converter falls back to non-CUDA behavior

Example configure command with CUDA raster enabled:
- `cmake -S . -B build_cuda_x64 -G Ninja -DGS2PC_ENABLE_CUDA_RASTER=ON`

## Example commands

From inside `cpp_native/`:

- configure:
  - `cmake -S . -B build -G Ninja`
- build:
  - `cmake --build build`
- show CLI help:
  - `build/gs2pc_cli.exe --help`
- run CUDA forward colour extraction on the bundled dataset layout:
  - `run_set_cuda_forward_test.cmd`

### Using `CMakePresets.json` in an IDE

`cpp_native/` now includes a `CMakePresets.json` file for IDE-friendly configuration.

Available configure presets:
- `x64-release`
- `x64-release-cuda`

Available build presets:
- `x64-release`
- `x64-release-cuda`

Recommended IDE workflow:
- open `cpp_native/` as the CMake project root
- select the `x64-release-cuda` preset when CUDA raster support is desired
- select the `x64-release` preset when a non-CUDA native build is preferred
- configure and build directly from the IDE using the selected preset

Equivalent command-line usage:
- `cmake --preset x64-release-cuda`
- `cmake --build --preset x64-release-cuda`

From the repository root with absolute or relative paths:

- `cmake -S 3DGS-to-PC/cpp_native -B 3DGS-to-PC/cpp_native/build -G Ninja`
- `cmake --build 3DGS-to-PC/cpp_native/build`

## Current practical status

At the time this note was written, the native application was successfully configured and built through CMake and Ninja, and the CLI help command executed successfully.

On a successful run, the CLI prints a stage-level timing breakdown:

```
Timing breakdown (ms): load_gaussians=<N>, load_cameras=<N>, forward=<N>, convert=<N>, export_total=<N>
Export I/O detail (ms): header=<N>, vertices=<N>, flush=<N>, total=<N>
Total processing time: <N> ms
```

The export format printed in the summary reflects the active `--export_format` setting (`binary` or `ascii`).

Updated: 2026-03-17
