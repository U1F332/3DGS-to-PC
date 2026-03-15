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

The current `cpp_native/` implementation is partially consistent with that original behavior, but it is not yet feature-equivalent.

### Areas that are already broadly aligned
- Both paths target conversion from Gaussian scene data to exported point clouds.
- Both paths accept Gaussian `.ply` and `.splat` style inputs.
- Both paths include support for COLMAP-derived camera data.
- Both paths expose controls related to point count, opacity filtering, bounding box filtering, SH-related settings, and visibility-oriented options.
- Both paths can write point-cloud PLY output.

### Areas where `cpp_native/` currently matches only at a structural or CLI level
- Some command-line options in `gs2pc_cli` mirror the Python interface, but not every option currently has the same execution depth behind it.
- `renderer_type`, colour-quality handling, transform-related options, and visibility-related options are represented in the native CLI, but the native path does not yet reproduce the full Python rendering workflow end to end.
- CUDA raster support in `cpp_native/` is currently focused on diagnostic `markVisible` and forward wrapper paths rather than full parity with the original Python rendering behavior.

### Areas where the original Python implementation is still more complete
- The Python path contains the main end-to-end workflow for rendered colour estimation through the repository's renderer integration.
- The Python path includes optional mesh generation and post-processing utilities.
- The Python path includes mask loading and downstream use patterns that are not yet fully reproduced in the native converter.
- The Python path contains more mature scene-processing behavior around visibility contribution, surface-distance filtering, and rendering-driven colour generation.
- The Python path depends on Torch-based tensor operations and renderer integration that are not yet fully replaced by the native code.

### Practical assessment
- Input compatibility: partial-to-good
- CLI surface compatibility: moderate
- Native loading and export coverage: good
- Rendering parity with the Python pipeline: limited
- Mesh and post-processing parity: not yet achieved
- Overall end-to-end feature parity: incomplete

### Conclusion

`cpp_native/` should currently be viewed as a working native scaffold with useful real functionality, not as a drop-in feature-complete replacement for `gauss_to_pc.py`.

It already covers important repository behavior such as native input parsing, COLMAP loading, Gaussian sampling, and PLY export, but the original Python implementation remains the more complete reference path for the full repository feature set.

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
Responsible for writing point clouds as ASCII PLY.

Current output behavior:
- always writes vertex positions
- writes normals when available
- writes RGB colour as `uchar`

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

Example configure command with CUDA diagnostics enabled:
- `cmake -S . -B build -G Ninja -DGS2PC_ENABLE_CUDA_RASTER=ON`

If CUDA is unavailable, the project still builds without the raster diagnostic integration.

## Example commands

From inside `cpp_native/`:

- configure:
  - `cmake -S . -B build -G Ninja`
- build:
  - `cmake --build build`
- show CLI help:
  - `build/gs2pc_cli.exe --help`

From the repository root with absolute or relative paths:

- `cmake -S 3DGS-to-PC/cpp_native -B 3DGS-to-PC/cpp_native/build -G Ninja`
- `cmake --build 3DGS-to-PC/cpp_native/build`

## Current practical status

At the time this note was written, the native application was successfully configured and built through CMake and Ninja, and the CLI help command executed successfully.

Updated: 2026-03-15
