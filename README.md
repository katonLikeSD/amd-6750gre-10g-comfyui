# AMD 6750 GRE 10G 显存本地生图/生视频流程跑通参考

> 在 **RX 6750 GRE 10G（gfx1031 / RDNA2 / ROCm）** 上，用 ComfyUI 跑通 **Wan2.2 TI2V-5B 视频生成**（含图片生成环境）的完整实测数据与踩坑记录。所有数字为本地实测（测试基础设施随仓库开源），非二手转述。

## TL;DR — 直接抄的最优参数

```bash
# 启动（start.sh）
export HSA_OVERRIDE_GFX_VERSION=10.3.0
cd ~/ComfyUI && python3 main.py --fp32-vae
```

| 项 | 值 |
|---|---|
| 分辨率 × 帧数 | **832×480 @ 33帧**（约 2 秒 @16fps） |
| KSampler | 18步 · cfg 5.0 · `dpmpp_2m_sde_heun` · `sgm_uniform` |
| VAEDecodeTiled | **tile_size=128 · overlap=32 · temporal_size=8 · temporal_overlap=4** |
| 文本编码器 | `umt5_xxl_fp8` 挂 **CPU**（CLIPLoader device=cpu） |
| 实测 | **244.5 秒/条**，采样峰值 **8.2GB / 10GB** |

⚠️ **VAEDecodeTiled 默认参数（512/64/64/8）在这张卡上 100% OOM**（单块 fp32 激活 6.33GB），必须改成上表。

---

## 一、完整环境（逐项版本）

### 硬件

| 部件 | 型号 / 规格 |
|---|---|
| GPU | **AMD Radeon RX 6750 GRE 10GB**（Navi 22 核心，`gfx1031`；lspci 误报 Navi 23 属正常） |
| 显存 | 10.0 GB（`/sys/class/drm/card1/device/mem_info_vram_total`） |
| CPU | AMD Ryzen 5 7500F（6核12线程，Zen4） |
| 内存 | 32 GB（测试时可用 27 GB）+ 2 GB swap |
| 磁盘 | NVMe，测试时剩余 20 GB |

### 操作系统与驱动

| 项 | 版本 |
|---|---|
| 发行版 | **Ubuntu 22.04.5 LTS** |
| 内核 | 6.8.0-138-generic（内核自带 amdgpu 驱动，无需 amdgpu-dkms） |
| ROCm 用户态 | rocm-core 7.2.4 · rocm-smi-lib 7.8.0 · libhsa-runtime64-1 5.0.0（Ubuntu 源） |
| **关键环境变量** | `HSA_OVERRIDE_GFX_VERSION=10.3.0`（gfx1031 不在 ROCm 官方支持列表，必须伪装成 gfx1030） |

### Python 与依赖（系统 Python，无 venv）

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.10.12 | 系统自带 |
| **torch** | **2.5.1+rocm6.2** | 官方 ROCm wheel（自带 HIP 运行时，与系统 rocm-core 版本解耦） |
| torchvision / torchaudio | 0.20.1 / 2.5.1+rocm6.2 | 同 wheel 集 |
| triton | 3.8.0 | pip 装的独立版，与 torch 2.5.1 不匹配 → **torch.compile 不可用** |
| numpy / pillow | 2.2.6 / 12.3.0 | |
| gguf | 0.19.0 | GGUF 加载 |
| av (PyAV) | 17.1.0 | **无 ffmpeg 二进制时的视频合成替代**（读帧/写 mp4） |
| transformers / accelerate | 5.16.1 / 1.15.0 | |
| einops / kornia / safetensors / scipy / aiohttp | 0.8.2 / 0.8.2 / 0.8.0 / 1.15.3 / 3.14.3 | ComfyUI 常规依赖 |

> 注意：**没有 xformers、没有 sage-attention、没有 flash-attention**——RDNA2 上这些要么不支持要么需重编译，实测 ComfyUI 自动走 PyTorch SDPA math 后端 + 内部分块（见"意外结论"）。

### ComfyUI

| 项 | 版本 |
|---|---|
| ComfyUI | **0.37.2**（commit `830232b8`，2026-09-23） |
| 前端 | comfyui_frontend_package **1.52.7** |
| comfy-kitchen | 0.2.28（本卡上 triton 后端不可用，自动回退） |
| 启动参数 | `--fp32-vae`（VAE 全精度；bf16 权衡见踩坑第 6 条） |

**自定义节点**（ComfyUI-Manager 安装；zip 安装的无 git 版本记录）：

| 节点包 | 用途 | 本仓库结论 |
|---|---|---|
| ComfyUI-GGUF | `UnetLoaderGGUF` 加载 GGUF 量化 UNet | ✅ 核心必需 |
| ComfyUI-Manager | 节点管理 | ✅ |
| ComfyUI-WanVideoWrapper | Wan 高级功能（FlashVSR 超分、context_windows 长视频） | ⚠️ 不支持直接加载 GGUF；FlashVSR 待验证 |
| ComfyUI-ControlOrder-FreeMemory | 执行顺序/显存释放控制 | 保留观察 |
| ComfyUI-AnyDeviceOffload | 跨设备卸载 | 保留观察 |
| tiledvaelite | `LTTiledVAEDecode` | ❌ 实测输给官方 VAEDecodeTiled |
| comfyui-qwen35-anima | Anima 动漫模型支持 | 图片侧 |
| ComfyUI-SA-Nodes-QQ | 社区节点包 | 未使用 |

---

## 二、模型清单（类型与规格，权重不分发）

### 视频生成（本仓库主角）

| 类型 | 文件 | 规格 | 说明 |
|---|---|---|---|
| 扩散模型（UNet） | `Wan2.2-TI2V-5B-Q4_K_M.gguf` | 3.2 GB，**GGUF Q4_K_M 量化**，5B 参数 | 文/图生视频一体（TI2V），Apache-2.0 官方权重的量化版 |
| 扩散模型（UNet） | `wan225bi2vspmixver021n_v21.gguf` | 3.6 GB，GGUF，5B | 社区混剪版（spmix），采样峰值 +0.35G |
| 文本编码器 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.4 GB，**fp8_e4m3fn** | umt5-xxl，挂 CPU 省显存 |
| VAE | `wan2.2_vae.safetensors` | 1.3 GB，fp16 权重 | **48 通道 latent、16×16×4 压缩**（与 2.1 的 16ch/8× 不通用！） |
| VAE 备选 | `DR34ML4Y_TI2V_5B_V1.safetensors` | 0.3 GB | 社区修复版 VAE（未深入测试） |
| 轻量 VAE | `taew2_1.pth` | 22 MB | 快速预览用 |

### 图片生成（环境保留，底模现状见备注）

| 类型 | 说明 |
|---|---|
| SDXL 系动漫 checkpoint（novaAnimeXL 等） | ⚠️ 当前硬盘上缺失（曾被清理），工作流断链；补回即可用 |
| Anima 动漫模型（qwen35-anima 节点 + Qwen3-0.6B 编码器 + qwen_image_vae） | 编码器/VAE 在位，checkpoint 缺失 |
| Flux2 Klein GGUF fp8 | 缺失 |
| clip_l / 各类风格 LoRA（SDXL/Illustrious 系，0.1-0.7 GB × 若干） | 在位（LoRA 属个人内容，不在本仓库示例中引用） |

**图片侧结论（基于 SDXL 经验+本机实测显存余量）**：fp16 SDXL 在 896×1152 内安全；1024×1536 需配合 tiled VAE 解码。

---

## 三、工作流（`workflows/` 可直接拖入 ComfyUI）

### 1. `wan22_t2v_832x480_33f.json` — 文生视频·日常最优档

节点链：`VAELoader → CLIPLoader(cpu) → CLIPTextEncode×2 → UnetLoaderGGUF → Wan22ImageToVideoLatent → KSampler → VAEDecodeTiled → SaveAnimatedWEBP`

关键参数：width=832 height=480 **length=33**（帧数必须 4n+1：9/17/33/49/65/81/121）；steps=18 cfg=5.0；VAEDecodeTiled 四项见 TL;DR。

### 2. `wan22_i2v_chain_segment.json` — 尾帧链式延长段

与 1 相同，外加 `LoadImage → Wan22ImageToVideoLatent.start_image`。用法：
1. 跑完一段，导出其**尾帧**为 PNG，放入 `ComfyUI/input/`（本仓库脚本：PIL 一行 `im.seek(n-1)` 即可）
2. 该帧作为下一段 `start_image`，生成 33 帧新段（实测峰值 9.3G，4.2 分钟/段）
3. N 段拼接 = 任意时长（PyAV 或剪辑软件）
4. 稳妥做法：每段 length=36（4n+1 取 33 或 37），裁掉尾部 3-4 帧（社区 wrapper 路径有末帧噪声坑，核心节点路径实测干净，裁掉更保险）

### 3. 直接长生成

`length=121`（5 秒@24fps）实测采样峰值 8.52G 通过；**>121 帧模型会"回弹"首帧**（社区反馈，未深测）。

---

## 四、踩坑大全（全部真实踩过）

### 显存/崩溃类

1. **VAEDecodeTiled 默认参数必炸**：512/64/64/8 时单个解码块（512px×64帧 fp32）激活 6.33GB → OOM。唯一可靠参数 `128/32/8/4`（25 组矩阵全过，峰值 2.7-3.4G）。
2. **untiled 解码直接崩进程**：`VAEDecode`（不切块）在 10G 卡上触发 ROCm 硬崩溃 `Fatal Python error: Aborted / GPU core dump failed`——不是 OOM 异常，ComfyUI 整个死掉。绝对别用。
3. **fp32 VAE 走低显存加载路径有非确定性**：同参数偶发 OOM（冷启动首测 8.31G 炸、重启后 6.27G 过）。遇到"必过参数突然炸"先重启 ComfyUI。
4. **ROCm 显存碎片污染**：阶梯测试里连续 OOM 尝试会让后续必过组合假 OOM（720P@81帧 保底参数在 4 次 OOM 后炸、全新状态 2.73G 轻松过）。测试脚本设计要每 N 次失败重启一次。
5. **`--fp32-vae` 的隐性代价**：VAE 权重 fp32 占 2.7GB（bf16 仅 1.35GB），且 ComfyUI 的显存预估公式（~10.8GB）强制 VAE 进 lowvram 部分加载模式——这就是默认 tile 参数在别处能用、这里炸的根因之一。

### 性能/画质类

6. **bf16 VAE 是权衡不是免费午餐**：输出与 fp32 逐像素一致、激活减半（256px/时序32 可过，4.92G），**但 gfx1031 的 MIOpen conv 路径下 bf16 解码慢 60%**（480P@81帧：199s→321s）。fp32+小tile 仍是默认推荐。
7. **512px tile 在 fp32/bf16 下都 OOM**；256px 是 bf16 的甜点位（反超 fp32 速度）。
8. **18 步有彩色斑块**（高频区域如草地），35 步干净；步数不影响显存只影响时间。
9. **tiledvaelite (LTTiledVAEDecode) 输给官方**：同规模 192.7s/7.26G vs 184.6s/4.92G。
10. **hipBLASLt 警告可忽略**：`Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas`——gfx1031 伪装 gfx1030 的正常回退，不影响结果。

### 测试方法类（开源脚本里都修掉了）

11. **裸 SDPA ≠ ComfyUI 实际行为**：独立 benchmark 里 `F.scaled_dot_product_attention` 在 8K token（batch1）就 OOM（math 后端物化 12·N²·4B 的 fp32 注意力矩阵），但 ComfyUI 实测 75K token 只占 ~8.5G——它对 query 做了分块。**别拿裸 torch 的结论吓退自己**。
12. **0.5s HTTP 轮询漏峰**：`/system_stats` 0.5s 采样把峰值读低 ~1.5GB；改 0.1s 直读 sysfs `mem_info_vram_used` 才是真峰值（driver.py 已实现）。
13. **ComfyUI 节点缓存会伪造"20 倍提速"**：重复提交完全相同的图（同参数同种子）会命中缓存秒回——曾据此误判 bf16 解码提速 20×，实为缓存命中。**测速必须换种子击穿缓存**。
14. **端口抢占**：`kill` 旧 ComfyUI 再启新的，若旧进程没死透，新进程会 `Port 8188 already in use` 静默失败，测试全跑在旧配置上——**切换配置后必须验证 `ss -ltn` 归属 + 进程 cmdline**（本仓库测试中就栽过一次，整轮"bf16 数据"实为 fp32）。
15. `pgrep -f "main.py"` 会匹配到执行它的 shell 自身命令行（自杀）；脚本里用 `pgrep -f "main[.]py"` 括号技巧规避。

### 环境类

16. **无 ffmpeg 二进制**：视频合成用已装的 PyAV（`av` 17.1.0）写 20 行脚本替代 VHS。
17. **triton 3.8.0 与 torch 2.5.1 不匹配** → torch.compile 路线关闭（升级 torch 又与 ROCm wheel 生态冲突，不值得）。
18. **RDNA2 无 flash/mem-efficient attention**：SDPA 落 math 后端，靠 ComfyUI 分块续命；不要尝试装 xformers（RDNA2 支持残缺）。

---

## 五、耗时曲线（18 步，实测锚点 ◆）

```
分钟
 30 ┤                                              ▲ 30.2(推算)
 20 ┤                                        ▲ 20.9
 12 ┤                                        ◆ 11.6
 10 ┤                            ▲      ● 9.6
  7 ┤                      ▲           ●
  4 ┤    ▲ 4.5        ◆ 4.1
  2 ┤         ● 2.0
  0 ┼────┬─────┬─────┬─────┬─────┬──→ 帧数
       17    33    49    65    81     ● 832×480  ▲ 1280×720
```

公式：`T ≈ 34s(文本编码,CPU) + 步数×步时 + 帧数×2.46s(480P解码)`；步时 ∝ token 数（480P@33帧 ≈ 8.6s，480P@81帧 ≈ 27.5s，121帧 ≈ 40s）。文本编码器挂 GPU 可省 20s/新提示词（编码期峰值 9.1G）。

## 六、目录结构与复现

```
├── README.md            本文件
├── docs/
│   ├── SUMMARY.md       完整测试报告（数据表）
│   └── deepseek-discussion.md  与 DeepSeek 三轮协作分析纪要
├── workflows/           可直接导入的工作流 JSON ×2
├── scripts/             driver.py（API 队列驱动+0.1s 显存监控+黑屏/NaN 检测）
│                        make_queue.py + 全部测试队列 jsonl
├── data/                原始测试结果 ~60 条（峰值显存/耗时/像素统计）
└── frames/              证据帧：花斑对比、链式衔接验证
```

```bash
# 复现单条：导入 workflows/wan22_t2v_832x480_33f.json → Queue Prompt，约 4 分钟
# 复现批量：python3 scripts/driver.py scripts/phase1_queue.jsonl
# 数据行格式：{id, status: ok/oom/error/timeout, wall_sec, peak_vram_gb,
#             video:{frames, mean_brightness, black, nan}}
```

## 声明

- 仓库内示例帧/结果为 AI 生成的普通素材（猫），不含敏感内容
- 模型权重因体积/版权不随仓库分发，仅列清单与获取指引（HuggingFace 搜对应文件名）
- 数据为单卡单次实测，不同驱动/内核/版本组合可能有 ±10% 波动；遇到问题先翻"踩坑大全"
