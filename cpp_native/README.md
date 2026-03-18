# `cpp_native` (native C++ implementation)

`cpp_native/` is the current native `C++20` pipeline for converting 3D Gaussian scenes (`.ply` / `.splat`) to point clouds.

It builds a CLI executable:
- `gs2pc_cli`

And an internal library:
- `gs2pc_core`

---

## What is current (important)

Compared with older notes, this native path is now focused on practical production usage:

- CUDA raster path is **enabled by default** (`GS2PC_ENABLE_CUDA_RASTER=ON`)
- C++ color extraction is aligned with the repository Python+CUDA behavior (precomputed color path)
- CUDA forward path now uses persistent GPU caches
- Per-Gaussian color/contribution statistics are accumulated on GPU and pulled back once
- PLY export is now **binary by default** (ASCII still available)
- CLI prints stage timing breakdowns
- Build automatically creates a portable `dist/` output directory

---

## Changes since `6792c6f0`

Major updates after `6792c6f0` include:

1. **Color alignment and parity improvements**
   - Native forward color path aligned to Python+CUDA output behavior.
   - Export channel conversion behavior adjusted for closer parity.

2. **CUDA raster performance work**
   - Persistent GPU caches for Gaussian static data and frame buffers.
   - GPU-side accumulation for best contribution color, total contribution, and min surface distance.
   - Reduced per-frame host transfers and synchronization overhead.

3. **Packaging / deployment flow**
   - Default CUDA-enabled build configuration.
   - Build now installs runtime dependencies into `dist/` automatically.

4. **Export and timing updates**
   - Default export format changed to binary PLY.
   - New `--export_format binary|ascii` option.
   - Detailed timing output for key stages and export I/O sub-stages.

---

## Build (important: use x64 Native Tools command prompt)

> No IDE UI is required, but you **must** use the correct x64 toolchain shell.

### Requirements

1. `CMake` >= 3.8
2. `Ninja`
3. Visual Studio 2022 C++ build tools
4. **Run commands in `x64 Native Tools Command Prompt for VS 2022`**
5. For CUDA build: CUDA Toolkit + compatible NVIDIA driver

If you do not use the x64 Native Tools shell, you may hit mixed-arch linker errors such as `LNK4272` (`x86` libraries linked into an `x64` target).

### Presets (current)

- `x64-release` = CUDA OFF (CPU-only build)
- `x64-release-cuda` = CUDA ON (native CUDA raster path enabled)

### CUDA ON vs CUDA OFF

#### `x64-release-cuda` (recommended)
- Enables native CUDA raster integration
- Uses GPU-based contribution/color extraction path
- Best parity and performance for the current pipeline

#### `x64-release`
- Builds without CUDA raster support
- CUDA diagnostic paths are unavailable (`--run_forward`, `--run_mark_visible`)
- Basic conversion still works, but color/performance path differs

### Command-line build (from `cpp_native/`)

#### CUDA ON

```bash
cmake --preset x64-release-cuda
cmake --build --preset x64-release-cuda
```

#### CUDA OFF

```bash
cmake --preset x64-release
cmake --build --preset x64-release
```

#### Clean rebuild (recommended when switching modes)

```bash
cmake --build --preset x64-release-cuda --clean-first
cmake --build --preset x64-release --clean-first
```

### Build output directories

- CUDA ON: `cpp_native/build_cuda_x64_release/dist/`
- CUDA OFF: `cpp_native/build_x64_release/dist/`

`dist/` contains `gs2pc_cli` and discovered runtime dependencies for portable execution.

---

## Output layout and portable build

After build, runtime files are installed into the active preset's `dist/` folder (for example `build_cuda_x64_release/dist` or `build_x64_release/dist`).

This includes:
- `gs2pc_cli` executable
- required runtime DLLs (where discoverable)
- CUDA runtime DLL copy for enabled CUDA builds (if found)

`gs2pc_dist` is part of default build (`ALL`), so normal build already prepares `dist/`.

---

## Basic usage

```bash
cpp_native/build/dist/gs2pc_cli --input_path <scene.ply|scene.splat> --output_path out.ply
```

With camera transforms for render-driven colors:

```bash
cpp_native/build/dist/gs2pc_cli \
  --input_path <scene.ply> \
  --transform_path <colmap_sparse_or_transforms> \
  --output_path out.ply
```

Choose export format:

```bash
--export_format binary   # default
--export_format ascii
```

Useful options:
- `--num_points <int>`
- `--colour_quality tiny|low|medium|high|ultra|original`
- `--visibility_threshold <float>`
- `--surface_distance_std <float>`
- `--mahalanobis_distance_std <float>`
- `--min_opacity <float>`
- `--cull_gaussian_sizes <float>`
- `--no_render_colours`
- `--quiet`

Full option list:

```bash
gs2pc_cli --help
```

---

## Timing output

Current CLI output includes:

- `load_gaussians`
- `load_cameras`
- `forward`
- `convert`
- `export_total`
- export I/O details: `header`, `vertices`, `flush`, `total`
- overall: `Total processing time`

This is intended to quickly locate bottlenecks (compute vs I/O).

---

## Current source layout

Headers:
- `include/gs2pc/config.h`
- `include/gs2pc/types.h`
- `include/gs2pc/io.h`
- `include/gs2pc/camera.h`
- `include/gs2pc/converter.h`
- `include/gs2pc/exporter.h`
- `include/gs2pc/raster.h`

Sources:
- `src/cli/cli.cpp`
- `src/cli/main.cpp`
- `src/io/io.cpp`
- `src/camera/camera_loader.cpp`
- `src/converter/converter.cpp`
- `src/exporter/exporter.cpp`
- `src/raster/raster.cpp`

---

## Notes

- Python remains the broader reference for all auxiliary features.
- Native main conversion path is now high-parity for practical Gaussian-to-point-cloud workflows, especially with CUDA enabled.
