# MiniMax H3 本地部署实录（6750 GRE 10G · Windows 原生 ROCm）

> 2026-10-14/15。在 **RX 6750 GRE 10G（gfx1031 / RDNA2）** 上从零部署 **MiniMax H3**（开源文/图生视频+音频模型）的完整过程与实测数据。所有数字为本地实测。

## TL;DR

| 结论 | 内容 |
|---|---|
| ✅ 可行档 | **576×320 @124帧（5s, 24fps, 含音频）**：8步Turbo LoRA **390 s/条**；4step Turbo LoRA **256 s/条**（比 6 步快 **1.52×**） |
| ❌ 不可行档 | 864×480：显存超限触发**权重抖动流式加载**，≈ **13 min/步**（已中断）→ 10G 卡甜点上限就在 576×320 附近 |
| 环境 | patientx-cfz/comfyui-rocm（**ROCm 10.1 + torch 2.14**）+ INT8 convrot 加速内核 + Adrenalin 26.8.1 + 124 GB 页面文件 |
| 模型 | Comfy-Org/MiniMax-H3 官方权重（ModelScope 镜像），7 文件 68 GB，**全部 SHA256 校验通过** |

> [!CAUTION]
> **博主工作流 JSON 存在断线缺陷**：`BasicScheduler.model` 与 `VAEDecodeAudio.samples` 两个输入引用了不指向自己的 link id（多输入共享同一 link 的序列化缺陷），导入后无法运行（报 `Required input is missing`）。修复：为这两个输入重建独立 link（见下"断线修复"节）。

## 一、环境（逐项）

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 6750 GRE 10GB（gfx1031，Navi 22，36 CU） |
| 驱动 | **Adrenalin 26.8.1**（32.0.21045.5002，2026-08-17；升级前是 2024-09 的 32.0.12011.1036） |
| 运行时 | comfyui-rocm 整合包自动装：ROCm 10.1 / torch 2.14.0+rocm10.1.0 / HIP 7.16.26385 |
| 加速内核 | ComfyUI-INT8-Fast-ROCM 全局劫持（native ops: int8_tensorwise / asym_w4a8_int8 / w6a8_int8 / **convrot_w4a4**） |
| 启动参数 | `--use-quad-cross-attention --disable-smart-memory --enable-dynamic-vram --cache-lru 20`（RDNA2 档） |
| 必要环境变量 | **`HSA_OVERRIDE_GFX_VERSION=10.3.0`**（RDNA1/2 必需，README 官方建议） |
| 虚拟内存 | D: 124 GB 初始 / 160 GB 最大（10G 卡跑 21GB 主模型的必需项） |
| 模型清单 | fl2va_int8_convrot 21G · ref2va_int8_convrot 21G · qwen3vl_32b_nvfp4_awq 15.7G · video_vae_fp16 5.2G · video_vae_int8 2.8G · audio_vae_fp32 0.6G · turbo 8step LoRA 2G |

## 二、实测数据（6/4/8 步对比）

| 分辨率 | 步数 | LoRA | 总耗时 | 采样 | 备注 |
|---|---|---|---|---|---|
| 384×256 | 6 | 8step | 270 s | 6 × 32.8 s | 冒烟测试 |
| 576×320 | 6 | 8step | 390 s | 6 × 32.8 s | **稳定可批量档** |
| 576×320 | 4 | **4step** | **256 s** | 4 × 47.8 s | 提速档（步数减半、单步变慢 46%，净提速 1.52×） |
| 864×480 | 8 | 8step | ~不可用 | **~13 min/步** | 显存超限 → 权重逐层流式重载抖动；已中断 |

- 首次跑某分辨率有内核编译开销（576×320 首步含 ~21 s；864×480 首步更高），之后重复跑同分辨率明显变快。
- 50.5 s/步 是 **576×320 @6步** 的稳态值（正式测试 390 s 总耗时含加载/解码）。

## 三、画质调优结论

1. **分辨率是最大杠杆**，但 10G 显存下 864×480 直接落入抖动区间 → 576×320 是甜点。
2. **步数**：8step LoRA 官方设计 8 步（博主工作流用 6 步）；4step LoRA 专为 4 步设计，实测 4×47.8 s，质量与 6 步档相当、速度 1.52×。
3. **编码质量**：SaveVideo 默认 `auto/auto` 码率很低（~0.85 Mbps）；改为 `mp4 + h264 + re-encode + crf 14` 显著减少压缩糊感。

## 四、断线修复（重要）

博主全部 13 个工作流 JSON 中，凡"同一输出接到多个输入"的位置都少写了 link 条目（如 `BasicScheduler(10).model` 和 `VAEDecodeAudio(14).samples` 分别引用了属于别人的 link 8 / link 13）。运行时被服务器校验拒绝：

```
Failed to validate prompt for output 16:
* BasicScheduler 10:  - Required input is missing: model
* VAEDecodeAudio 14:  - Required input is missing: samples
```

**修复方法**：对每个输入端，从其引用 link 的**源节点/源槽**重建一条指向本输入的新 link，重排 link id，并同步 `last_link_id`。本仓库 `workflows/` 内的 H3 工作流已全部修复版。

## 五、复现

```bat
:: 启动（整合包根目录）
set HSA_OVERRIDE_GFX_VERSION=10.3.0
comfyui-user.bat
```

- 模型：ModelScope `Comfy-Org/MiniMax-H3`（SHA256 清单随仓库）
- 工作流：`workflows/h3_t2v_576x320_6step_api.json`（8step LoRA 6步稳定档）· `workflows/h3_dance_4step_api.json`（4step 提速档）
- 环境完整构建：patientx-cfz/comfyui-rocm 一键脚本 + Adrenalin 26.8.1 + 124 GB 页面文件

---
*2026-10-15 凌晨：ZCode 自动化部署 + 测试 + 出片*
