# Hummingbird-XT 调研与可用性评估（AMD 官方）

> 2026-10-12。用户在社区看到 "Wan 2.2 + Hummingbird-XT"，本文是**核实结论**（数据源：GitHub API + HuggingFace API 直查，非转述）。

## 一、它是什么（已核实）

- **AMD 官方项目**：`github.com/AMD-AGI/HummingbirdXT`（权重：`huggingface.co/amd/HummingbirdXT`）
- **本质**：基于 **Wan2.2-5B** 的**步数蒸馏模型族 + 配套加速推理管线**（不是 LoRA、不是采样器补丁）
- **附带**：**轻量 VAE 解码器**（3D DW Conv 版）——官方称 **解码 14× 加速、显存 4× 降低**
- **另有 HummingbirdXTX**：基于 Wan2.1-1.3B 的**自回归长视频**模型（长片方向）
- **官方优化目标**：**Navi48（RDNA4）dGPU 与 Strix Halo iGPU**

### 官方性能表（121 帧 @704×1280）

| 模型 | MI300X | MI325X | Strix Halo | Navi48 |
|---|---|---|---|---|
| 原版 Wan-2.2-5B | 193.4 s | 153.9 s | 15000 s | **OOM** |
| Hummingbird-XT | 6.5 s | 3.8 s | 460 s | **36.4 s** |

## 二、对这台机器（6750 GRE 10G / RDNA2 gfx1031）的可用性

| 部件 | 大小 | 可用性判定 |
|---|---|---|
| 主 DiT（fp16） | 9.5 GB | ❌ 10G 显存放不下；磁盘也需先腾空间（本机常剩 ~8 GB） |
| 文本编码器 | 10.8 GB | ❌ 不需要（本机已有 umt5 fp8） |
| XTX 长视频 1.3B（DMD） | 5.4 GB | ⚠️ 理论可放，但走他们自己的管线（FastVideo fork） |
| **轻量 VAE 解码器** | **145 MB / 314 MB** | ⚠️ **不能直接用于 ComfyUI**（见下） |

**轻量 VAE 为什么不能直接用**：权重是新架构——`mid_block.resnets.*.depthwise_conv.weight [512,1,5,5,5]`（3D DW Conv），
共 106/134 个张量；而官方 `wan2.2_vae.safetensors` 是 196 张量的旧架构（`decoder.conv1.*`）。
**层结构不同 → ComfyUI 的 VAE 类无法加载**。（已下载留档：`models/vae/wan22_v3_tiling16_12.safetensors`、
`wan22_v1_tiling16_12.safetensors`）

## 三、结论与后续路线

1. **主线（整套 Hummingbird-XT）**：官方目标是 RDNA4/Strix Halo，本机 RDNA2 属**未支持配置**；且主权重 9.5GB
   放不进 10G 显存 → **不建议投入**。
2. **最有价值的部分 = 轻量 VAE 解码器**（直接打本机"解码 2.5–5 s/帧"的瓶颈）。要用它需要**移植**：
   参考实现在他们仓库的 FastVideo fork（`.../fastvideo/models/vaes/wanvae.py`），
   做成一个"只用解码器 + 他们那套 tiling（`t_block_size 16 / t_stride 12`）"的 ComfyUI 自定义节点。
   工作量：数小时级；风险：RDNA2 上 3D DW Conv 的实际加速幅度未知（需实测）。
   **列为候选项目**（收益：解码 14× / 显存 4×，直击当前最大瓶颈）。
3. 与 Turbo LoRA 的关系：**替代**而非叠加（都是蒸馏路线）。

## 四、参考

- 仓库：https://github.com/AMD-AGI/HummingbirdXT ｜ 权重：https://huggingface.co/amd/HummingbirdXT
- 官方博客：https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html
- 注意区分：`AMD-AGI/Hummingbird`（0.9B 文生视频，**无关项目**）