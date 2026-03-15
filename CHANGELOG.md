# CHANGELOG — 3DGS-to-PC (简洁说明)

概述
- 将仓库核心从 Python/Torch 向原生 C++/CUDA（`cpp_native/`）迁移的工作已进行。此变更集整理并保留了实际代码改动，清理了临时与迁移文档。

主要改动
- 新增：`cpp_native/` 原生实现骨架（包含 `include/gs2pc/*.h`、`src/*/*.cpp`、CMake 配置与最小 CLI）。
- 新增：原生 `markVisible` 与 `forward` 封装，可使用 COLMAP 相机驱动进行可见性与贡献统计。
- 改进：支持读取真实 Gaussian PLY / SPLAT 输入字段（位置、尺度、旋转、不透明度、SH 等）；支持 COLMAP 相机文件（`.bin`/`.txt`）。
- 改进：增加 `visibility_threshold`、`surface_distance_std` 等筛选策略，并可按 `--num_points` / `--exact_num_points` 导出点云。
- 文档：移除多份临时/迁移用文档（已整合为本 `CHANGELOG.md`）。

清理
- 删除了若干迁移说明文件与命令输出残留文件（这些文件为临时草稿或误保存的命令输出）。
- 保留并提交 `cpp_native/build*`（你指定保留 build 输出）。

注意事项与后续工作
- 本次提交将包含大量原生源文件与部分构建产物。请在推送远端前确认是否需要将二进制/构建文件加入版本控制（通常不推荐，但按你要求保留）。
- 建议后续添加或更新 `.gitignore` 以明确哪些构建产物应保留或忽略。
- 若需要，我可以：
  - 生成推荐的 `.gitignore` 并提交；
  - 将 build 文件移动到单独的目录并添加说明；
  - 或按你的要求把 build 文件从提交中排除。

简短历史记录（要点）
- 实现了从高斯输入到点云导出的纯 C++ 最小链路，包含相机加载、可见性标记、forward 聚合与点云导出。

---

生成者：仓库维护整理（自动化摘要）
日期：2026-03-15
