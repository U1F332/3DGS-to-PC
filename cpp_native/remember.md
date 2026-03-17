# 本次会话记录（防止记忆丢失）

## 目标与结论
- 目标：对齐 `python+cuda` 与 `cpp_native` 的颜色输出，并开始做性能优化。
- 结论：颜色不一致的主因是两边 CUDA 输入策略不同（Python 主流程走 `colors_precomp`，C++ 之前优先走 SH）。

## 已完成改动

### 1) 颜色对齐（已完成）
- 文件：`cpp_native/src/raster/raster.cpp`
  - C++ CUDA forward 改为对齐 Python 主流程：使用 `colors_precomp`，不再优先 SH 输入。
- 文件：`cpp_native/src/exporter/exporter.cpp`
  - 颜色量化由“四舍五入”改为“截断”，更接近 Python `astype(np.uint8)`。

### 2) 构建与打包（已完成）
- 文件：`cpp_native/CMakeLists.txt`
  - 默认开启 CUDA：`GS2PC_ENABLE_CUDA_RASTER=ON`。
  - 增加并启用默认自动打包到 `dist`（普通 build 即触发 install 到 dist）。

### 3) 性能优化第 1/2 轮（已完成）
- 文件：`cpp_native/src/raster/raster.cpp`
  - 新增高斯静态数据 GPU 缓存：`means/colors/opacities/cov3d` 跨相机复用。
  - 新增 forward 帧缓冲缓存：`out_color/out_depth/out_invdepth/radii/gauss_*` 与 `geom/binning/image` 复用。
  - 目标：减少每帧 `cudaMalloc/cudaFree` 与重复大块上传。

## 已提交 Commit
- `bee8c0b`
  - `Align C++ color path with Python CUDA and enable default dist packaging`
  - 主要是颜色对齐 + 默认 CUDA + 打包流程。
- `e177072`
  - `Optimize CUDA raster path with persistent Gaussian and frame buffer caches`
  - 主要是 raster 缓存化性能优化。

## 运行与发布状态
- 本地提交已完成。
- 曾因网络问题 push 失败；后续可在代理环境下重试推送。

## 仍可继续的优化（未做）
1. 将“最佳贡献颜色更新”下沉到 GPU，减少每帧整图 D2H 回传。
2. stream 异步流水与双缓冲。
3. pinned memory 优化 H2D/D2H。

## 备注
- 当前进入“冻结功能更新，优先速度优化”阶段。
- 用户已确认：本轮先到这里，不继续第三轮优化。
