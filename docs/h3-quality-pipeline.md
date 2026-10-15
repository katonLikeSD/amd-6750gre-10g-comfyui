# H3 画质增强流水线（超分 + 防抖 + 插帧）—— 10G 卡实测

> 2026-10-15 晚。针对"H3 输出人脸糊"的问题，在 576×320 源片上搭建并实测了逐帧修复流水线。所有数字为本地实测。

## 结论（TL;DR）

| 环节 | 方案 | 实测效果 |
|---|---|---|
| **超分** | 4x 模型逐帧 ×4 再降采样到 ×2（1152×640） | 三种模型都大幅改善；**UltraSharp 最锐 / Remacri 最均衡 / RealESRGAN-x4plus 最自然** |
| **防抖** | 时域 EMA 平滑（α=0.5，前后帧各 25%） | 帧间差 **1.68 → 1.33（−21%）**，波动 σ **0.59 → 0.31（−47%）** |
| **顺序** | 先 EMA 再超分 | 超分会放大帧间噪声（1.68→1.90+），必须先平滑再超分 |
| **插帧** | RIFE 124→248 帧 @48fps | 运动更顺滑（静止场景提升有限，动作场景收益大） |
| 编码 | h264 crf14 + 原音轨直通 | 文件约 ×4 大，脸/发丝细节无损保留到编码层 |

## 流水线（可复现）

```
H3 输出 mp4（576×320）
  → PyAV 抽帧（124 帧 PNG）
  → 时域 EMA 平滑（α=0.5）
  → 超分 ×4（UltraSharp / Remacri / RealESRGAN-x4plus）→ 降采样 ×2
  → （可选）RIFE 插帧 ×2 → 48fps
  → PyAV 编码 h264 crf14 + 原音轨 → 成品 mp4
```

## 模型选择建议

| 模型 | 特点 | 来源 |
|---|---|---|
| 4x_foolhardy_Remacri | 人脸自然与细节平衡最好（推荐主力） | philz1337x/upscaler（HF 镜像可下） |
| 4x-UltraSharp | 最锐，纹理/发丝细节最强，轻微锐化感 | Kim2091/UltraSharp |
| RealESRGAN-x4plus | 最自然柔和（内置 ncnn 版） | xinntao/Real-ESRGAN |

*人脸专用修复（CodeFormer/GFPGAN）有逐帧闪烁问题，须配合 EMA 使用 —— 待测。*

## 源头防抖（生成阶段，来自与 DeepSeek 的协作结论）

1. **禁用抽象运镜词**（zoom/orbit 等会被降级为随机缩放 → 抖源）；用 "static camera, locked-off shot"
2. **Pre-Act Lead-In**：不要光写 "standing still"，写 "takes a subtle breath, shifts weight slightly, then holds still"
3. 单镜头静止场景优先低运动、固定机位

## 适配本机

- 超分推理用 spandrel（torch/ROCm）或 ncnn-vulkan 均可，10G 卡 576×320→×4 单帧约 1–4 秒
- 工具与脚本在本机 D:\_dl_tmp（extract/EMA/encode/RIFE 全链路脚本化）

---
*2026-10-15，ZCode 自动化实验 + DeepSeek 协作*

## 附加实测（补充）

**提示词防抖（DeepSeek 建议验证）**：同一场景 seed，改进提示词（"static camera, locked-off shot" + "takes a subtle breath, shifts weight slightly, then holds still" — Pre-Act Lead-In，且完全不用抽象运镜词）→ 帧间差 **1.68 → 1.50（−11%）**。

**RIFE 插帧实测**：rife-v4.6（ncnn-vulkan，直接在 6750 GRE 上跑）124→248 帧 = 48fps 顺滑版；配合 ×2 超分的"终极版"已产出。

**本次产出（都在 output/video/）**：
| 文件 | 内容 |
|---|---|
| H3_static_dress_00001_.mp4 | 原始 576×320 |
| H3_static_up2x_ncnn.mp4 | ncnn x4plus 超分 ×2 |
| H3_static_up2x_ultra.mp4 | UltraSharp 超分 ×2 |
| H3_static_up2x_remacri.mp4 | Remacri 超分 ×2 |
| H3_static_ema_smooth48.mp4 | EMA 防抖 + RIFE 48fps |
| H3_static_ultimate_2x_48fps.mp4 | EMA+超分+RIFE 全链路终极版 |
| H3_static_v2_00001_.mp4 | 防抖提示词版 |

**工具链（本机可复现）**：D:\_dl_tmp\ 下 dump_frames.py / temporal_smooth.py（EMA+指标）/ encode_with_audio.py（crf14+音轨直通）/ chain_ultimate.sh 全链路；ncnn 工具在 tools/（注意：ncnn-vulkan 工具要求输出目录预先存在）。

## 特写构图实验（根治"脸糊"的源头手段）

帧内脸部像素预算对比（同为 576×320 输出）：
- 全身构图：脸约 **60×60 px** → 天然糊
- 特写构图：脸约 **200×200 px** → 原生清晰度大幅提升

**结论：构图 = 免费的分辨率。** 需要看脸就用特写构图（配合 static camera + locked-off），全身构图交给超分流水线补救。

## DeepSeek 第二轮建议（2026-10-15，待试清单）

| 优先级 | 方案 | 说明 |
|---|---|---|
| P0 | H3-Optimizations 节点包（H3 Memory Optimization / Sparse Attention） | 降生成峰值显存 |
| P0 | 超分后加 Deflicker 节点 | 残余闪烁兜底 |
| P1 | SeedVR2 3B + BlockSwap=16 起步（本机已装节点+运行） | 视频原生超分 |
| P1 | FlashVSR Tiny-Long（8-12G，124帧 2x 约 56s） | 速度优先超分 |
| P2 | XB_ToolBox 显存管理 | 系统级 |
| P2 | H3 Acc LoRA / LongLive-Plug（4步无CFG蒸馏） | 采样效率 |

**蒸馏模型认知（DS 确认）**：H3 引导蒸馏模型 guidance=1.0，**不可加 CFG**；蒸馏轨迹不能靠加步数扩展。

## SeedVR2 本地化进度（2026-10-15 晚）

- 节点：numz/ComfyUI-SeedVR2_VideoUpscaler ✓ 已装（依赖逐项 --no-deps 安装，ROCm torch 未被破坏）
- 模型：seedvr2_ema_3b-Q8_0.gguf（3.5G）+ ema_vae_fp16.safetensors → models/SEEDVR2/
- 参数：blocks_to_swap=16 / swap_io_components / offload cpu / VAE tiled 512 / resolution 640 (×2) / batch 5 (4n+1) / sdpa 注意力
- 工作流：D:\_dl_tmp\h3_seedvr2_test.json

## 本地化踩坑（SeedVR2 模型下载）

- hf-mirror 现走 **XET 存储**（cas-bridge.xethub.hf.co）：`curl -I` 拿到的是重定向页长度，**必须 `-IL` 跟随重定向**才拿到真实 Content-Length
- 单线程 hf-mirror 约 1-2.5MB/s；**4-6 线并行分段（curl -r）聚合 4-5MB/s**，3.6GB 模型约 14 分钟
- ComfyUI-Manager 在无代理时会报 raw.githubusercontent 连接失败（无害，本地节点照常）
