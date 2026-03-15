# Changelog

## Current repository status

This repository currently contains two main conversion paths for turning 3D Gaussian Splatting data into point clouds:

- A Python workflow centered around `gauss_to_pc.py` and its supporting loaders, render helpers, and mesh utilities.
- A native C++20 scaffold in `cpp_native/` with a small CLI, Gaussian loaders, COLMAP camera loading, point-cloud conversion, PLY export, and optional CUDA raster diagnostic hooks.

## What is present in the repository

### Python pipeline

- `gauss_to_pc.py` is the main Python entry point for generating dense point clouds from Gaussian scene files.
- The Python flow supports loading Gaussian data from `.ply` and `.splat` inputs.
- Camera and transform data can be loaded through `transform_dataloader.py`.
- Optional mask loading is provided by `mask_dataloader.py`.
- Rendering integration is handled through `gauss_render.py` and camera helpers in `camera_handler.py`.
- Gaussian parsing and PLY export helpers are implemented in `gauss_dataloader.py`.
- Optional mesh-related processing is provided by `mesh_handler.py`.

### Native C++ pipeline

- `cpp_native/` builds a C++20 native implementation with CMake.
- The native target defines a reusable core library `gs2pc_core` and a CLI executable `gs2pc_cli`.
- The CLI accepts conversion-oriented arguments such as input path, output path, transform path, point count, SH degree, visibility threshold, opacity filtering, bounding box filtering, and colour quality.
- Native Gaussian loading supports:
  - binary `.splat` input
  - ASCII and binary little-endian `.ply` input
  - detection of point-cloud PLY versus Gaussian PLY layouts
- The native loader reads practical Gaussian attributes from repository-supported inputs, including:
  - position
  - scale
  - rotation
  - opacity
  - RGB colour
  - normals when present
  - SH coefficients when present in PLY input
- Native camera loading supports COLMAP text and binary camera/image files.
- Native conversion can filter by opacity, bounding box, and Gaussian size culling ratio.
- Native point generation distributes samples across Gaussians and exports ASCII PLY point clouds.
- Native export preserves normals when available.

### CUDA-related native diagnostics

- `cpp_native/` includes optional CUDA raster integration controlled by `GS2PC_ENABLE_CUDA_RASTER`.
- When enabled and available, the native CLI can run:
  - `markVisible` diagnostics
  - forward raster diagnostics
- These paths are implemented as optional wrappers and the build falls back cleanly when CUDA raster support is not enabled.

## Build configuration in the repository

- The native code uses CMake with a required minimum version of `3.8`.
- The native project is configured for `C++20`.
- CUDA raster support is optional rather than mandatory.
- A `clang-format` target is generated when `clang-format` is available.

## Notes

- This changelog was rewritten in English to match the repository's current checked-in contents rather than an earlier migration summary.
- It describes the actual code and build structure currently present in the workspace.

---

Updated: 2026-03-15
