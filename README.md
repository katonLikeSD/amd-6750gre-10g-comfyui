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
| 输出 | **`.webp` 动图**（`SaveAnimatedWEBP`，16fps）——**不是 mp4**，见第六节说明 |

> [!CAUTION]
> **VAEDecodeTiled 默认参数（512/64/64/8）在这张卡上必 OOM**（单个解码块 fp32 激活 6.33GB；本机 25 组矩阵全炸，未发现例外）。导入任何工作流后第一件事：把 tile 四项改成 `128 / 32 / 8 / 4`。

> [!CAUTION]
> **不要用 `VAEDecode`（untiled 整段解码）**——它不是 OOM 报错，而是直接触发 ROCm 硬崩溃（`Fatal Python error: Aborted / GPU core dump failed`），ComfyUI 整个进程死掉。

> [!CAUTION]
> **装自定义节点时警惕 ComfyUI-Manager 把 torch 换成 CUDA 版**——直接毁掉 ROCm 环境（本机真实发生）。用 Manager 装节点前先备份 `torch*`，装后必查 `python3 -c "import torch;print(torch.version.hip)"` 是否为 `6.2.x`。详见踩坑 5。

> [!WARNING]
> **遇到"必过参数突然 OOM"先重启 ComfyUI 再降参数**——ROCm 显存碎片会污染后续测试（连续几次 OOM 后，正常参数也会假 OOM）。

> [!TIP]
> **SDXL 图片生成（novaAnimeXL，2026-10-07 补回实测，18 次生成）**：1024×1024 **15 步 ≈ 20s/张**（画质干净）、30 步 ≈ 40s；稳态 ≈1.17s/it 对采样器/cfg 不敏感；untiled 解码实测上限 **1536×1536（峰值 9.15GB）**——顶部两条 CAUTION 只针对 Wan 视频 VAE，别给 SDXL 套 tile 参数。**避开 1216×1824/1824×1216 bucket：本机必出纯黑（静默失败）**。详见 [docs/sdxl-novaanimexl-test.md](docs/sdxl-novaanimexl-test.md)。

---

## 一、完整环境（逐项版本）

### 硬件

| 部件 | 型号 / 规格 |
|---|---|
| GPU | **AMD Radeon RX 6750 GRE 10GB**（Navi 22 核心，`gfx1031`；lspci 误报 Navi 23 属正常） |
| 显存 | 10.0 GB（`/sys/class/drm/card1/device/mem_info_vram_total`） |
| CPU | AMD Ryzen 5 7500F（6核12线程，Zen4） |
| 内存 | 32 GB（测试时可用 27 GB）+ 2 GB swap |
| 磁盘 | NVMe；**最低空间要求 ~30GB**（环境约 5G + 视频模型约 16G + 输出缓冲，见模型清单） |

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
| **torch** | **2.5.1+rocm6.2** | 官方 ROCm wheel（自带 HIP 运行时，与系统 rocm-core 解耦） |
| torchvision / torchaudio | 0.20.1 / 2.5.1+rocm6.2 | 同 wheel 集 |
| triton | 3.8.0 | 与 torch 2.5.1 **不匹配** → torch.compile 不可用（见"已知不可用"） |
| numpy / pillow | 2.2.6 / 12.3.0 | |
| gguf | 0.19.0 | GGUF 加载 |
| av (PyAV) | 17.1.0 | **无 ffmpeg 二进制时的视频合成替代**（读帧/写 mp4） |
| transformers / accelerate | 5.16.1 / 1.15.0 | |
| einops / kornia / safetensors / scipy / aiohttp | 0.8.2 / 0.8.2 / 0.8.0 / 1.15.3 / 3.14.3 | ComfyUI 常规依赖 |
| comfy-kitchen | 0.2.28 | triton 后端自动禁用，回退正常 |

> 注意：**没有 xformers、没有 sage-attention、没有 flash-attention**——RDNA2 上这些要么不支持要么需重编译，实测 ComfyUI 自动走 PyTorch SDPA math 后端 + 内部分块（见"意外结论"）。

### ComfyUI

| 项 | 版本 |
|---|---|
| ComfyUI | **0.37.2**（commit `830232b8`，2026-09-23） |
| 前端 | comfyui_frontend_package **1.52.7** |
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

### 安装命令（从零复现本环境）

```bash
# 0. ROCm 用户态（复现者若系统无 ROCm，第一步就会卡住；本机为 rocm-core 7.2.4）
#    方式 A（官方 amdgpu-install，推荐）:
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/jammy/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.4.70204-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms      # --no-dkms: 内核自带 amdgpu 驱动够用
#    验证: rocminfo | grep -i gfx   → 应出现 gfx1031
#    加入用户组（否则访问 /dev/kfd 被拒）:
sudo usermod -aG render,video $USER && newgrp render

# 1. 系统依赖 (Ubuntu 22.04)
sudo apt install -y python3-pip git

# 2. PyTorch ROCm 轮子（与 torch 2.5.1 对应的索引）
pip3 install torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 torchaudio==2.5.1+rocm6.2 \
  --index-url https://download.pytorch.org/whl/rocm6.2
#    国内走 Aliyun 镜像更快（本机实际用的这个）:
#    pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
#      -f https://mirrors.aliyun.com/pytorch-wheels/rocm6.2/ --no-cache-dir

# 3. ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
cd ~/ComfyUI && pip3 install -r requirements.txt
pip3 install comfyui-frontend-package==1.52.7   # 如默认前端版本不匹配

# 4. 必需的自定义节点
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/ltdrdata/ComfyUI-Manager   # 用前必读踩坑 5，防它换 torch

# 5. 启动（gfx1031 必须带 HSA 覆盖）
export HSA_OVERRIDE_GFX_VERSION=10.3.0
cd ~/ComfyUI && python3 main.py --fp32-vae
```

### ✅ 环境验证清单（装完逐项打勾，省得跑到一半才发现问题）

```bash
# 1. GPU 是否认到（应输出 gfx1031）
rocminfo | grep -i gfx

# 2. PyTorch ROCm 是否可用（两行都要过：True + hip 6.2.x）
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
#   → True 6.2.41133-xxx  ✅   |   False None  ❌ 被换成 CPU/CUDA 版，见踩坑 5

# 3. 显存总额是否 10G（确认没被别的进程占）
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool | grep -i vram_total

# 4. 端口属于当前这个 ComfyUI 进程（防踩坑 16 的端口抢占）
ss -ltnp | grep 8188
```

**这三步过了，基本环境就稳了。** 再跑一次 TL;DR 里的视频参数，若在 240-250s 出片、峰值 8G 左右，即视为环境复现成功（时序类数字受驱动/内核影响，允许 ±10%）。

---

## 二、模型清单与下载源

### 视频生成（本仓库主角）

| 类型 | 文件 | 规格 | 放置目录 | 来源（可直接复制；下载前以仓库实际文件为准） |
|---|---|---|---|---|
| 扩散模型 UNet | `Wan2.2-TI2V-5B-Q4_K_M.gguf` | 3.2 GB，**GGUF Q4_K_M 量化**，5B | `models/unet/` | HF: `QuantStack/Wan2.2-TI2V-5B-GGUF` · 国内: `hf-mirror.com/QuantStack/Wan2.2-TI2V-5B-GGUF` |
| 扩散模型 UNet | `wan225bi2vspmixver021n_v21.gguf` | 3.6 GB，GGUF，5B 社区混剪版 | `models/unet/` | Civitai 搜 `wan225bi2vspmix`（社区版，非官方） |
| 文本编码器 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.4 GB，fp8_e4m3fn | `models/text_encoders/` | HF: `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`（`split_files/text_encoders/` 子目录） |
| VAE | `wan2.2_vae.safetensors` | 1.3 GB fp16，**48 通道 latent、16×16×4 压缩**（与 2.1 的 16ch/8× 不通用！） | `models/vae/` | 同上 `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`（`split_files/vae/`） |
| VAE 备选 | `DR34ML4Y_TI2V_5B_V1.safetensors` | 0.3 GB 社区修复版 | `models/vae/` | Civitai 搜 `DR34ML4Y`（未深入测试） |
| 轻量 VAE | `taew2_1.pth` | 22 MB 快速预览 | `models/vae_approx/` | 同上 Comfy-Org 仓库 |

下载加速（本机实测可用的国内通道）：

```bash
# HuggingFace 走 hf-mirror 镜像（本机实际用这个，直连常断）
export HF_ENDPOINT=https://hf-mirror.com
hf download QuantStack/Wan2.2-TI2V-5B-GGUF Wan2.2-TI2V-5B-Q4_K_M.gguf --local-dir ~/ComfyUI/models/unet
# Comfy-Org 整套（含 umt5 + vae）也可用 hf-mirror 前缀如上
```
> 注：本机实测 GitHub 网页/CLI 直连不稳（经常 EOF/超时），但 `api.github.com` 稳定、`hf-mirror.com` 可用；大文件下载用 `curl -C -` 或 `hf download` 支持续传，别用会截断的直连。

### 图片生成（环境保留，底模现状如实标注）

| 类型 | 说明 |
|---|---|
| SDXL 系动漫 checkpoint | ✅ 已补回并实测（2026-10-07，18 次生成）：`novaAnimeXL_ilV190.safetensors`（6.9GB，自带双 CLIP+VAE，放 `models/checkpoints/`），15 步 20s / 30 步 40s @1024²，见 [docs/sdxl-novaanimexl-test.md](docs/sdxl-novaanimexl-test.md)。下载源：Civitai 搜 `novaAnimeXL`（选 illustrative/IL 版本） |
| Anima 动漫（qwen35-anima 节点 + Qwen3-0.6B 编码器 + qwen_image_vae） | 编码器/VAE 在位，checkpoint 缺失 |
| Flux2 Klein GGUF fp8 | 缺失 |
| clip_l / 各类风格 LoRA（SDXL/Illustrious 系 × 若干） | 在位（LoRA 属个人内容，不在示例中引用） |

#### ⚠️ GGUF 模型不带 LoRA 加载能力（如实标注）

Wan2.2 用的是 GGUF（`UnetLoaderGGUF`），**ComfyUI 原生 GGUF 加载节点不支持挂 LoRA**（LoRA 对量化权重做 merge 需要原始精度权重）。要用 LoRA 得走社区节点（如 `ComfyUI-GGUF` 的 `UnetLoaderGGUFAdvanced` 或 WanVideoWrapper），或者换非量化 safetensors 底模。**本仓库未实测过 GGUF+LoRA 组合**，不做效果承诺——标在这里是提醒别默认它会像 safetensors 那样直接接 `LoraLoader`。

图片侧结论（**已实测，2026-10-07，novaAnimeXL，18 次生成**）：稳态 ≈1.17s/it @1MP（dpmpp_2m / dpmpp_2m_sde / euler_ancestral 同速，cfg 5/7 同速）；1024² 15步≈20s、30步≈40s；untiled `VAEDecode` 实测到 1536×1536 正常（峰值 9.15GB），1024×1536/1152×1728 均过；batch2 @1024² 安全（8.15GB）。**例外：1216×1824 与 1824×1216 两个标准 bucket 本机必出纯黑**（形状级静默失败，换种子/采样器/tiled 解码均无效，规避用 1152×1728）。原先"896×1152 内安全 / 1024×1536 需 tiled"的经验预测已被实测数据替换，完整矩阵见 [docs/sdxl-novaanimexl-test.md](docs/sdxl-novaanimexl-test.md)。

---

## 三、工作流（`workflows/` 可直接拖入 ComfyUI）

### 1. `wan22_t2v_832x480_33f.json` — 文生视频·日常最优档

节点链（导入后按此定位参数）：

```
VAELoader(wan2.2_vae) ─┬─→ Wan22ImageToVideoLatent ──┐
                       │    width=832 height=480      │
CLIPLoader(umt5,       │    length=33 ←帧数(4n+1)     │
  device=cpu!) ─→ CLIPTextEncode×2(正/负提示词) ──→ KSampler ──→ VAEDecodeTiled ──→ SaveAnimatedWEBP
UnetLoaderGGUF(Q4_K_M)─┘    steps=18 cfg=5.0          128/32/8/4 ←必改!
                            dpmpp_2m_sde_heun
                            sgm_uniform
```

关键参数三处：**① CLIPLoader 的 device 选 `cpu`**；**② VAEDecodeTiled 四项 = 128/32/8/4**；**③ length 必须 4n+1**（9/17/33/49/65/81/121）。

输出是 **`.webp` 动图**（`SaveAnimatedWEBP`），不是 mp4——详见第六节"输出格式说明"。

### 2. `wan22_i2v_chain_segment.json` — 尾帧链式 I2V 延长段

与 1 相同，外加 `LoadImage → Wan22ImageToVideoLatent.start_image`。用法：
1. 跑完一段，导出其**尾帧**为 PNG 放入 `ComfyUI/input/`（PIL 一行：`im.seek(im.n_frames-1)`）
2. 该帧作为下一段 `start_image`，生成新 33 帧段（实测峰值 9.3G，4.2 分钟/段）
3. N 段拼接 = 任意时长（PyAV 或剪辑软件）
4. 稳妥做法：每段多生成 3-4 帧并裁掉尾部（社区 wrapper 路径有"末帧噪声"报告；核心节点路径实测干净，裁掉更保险）

### 3. 直接长生成

`length=121`（5 秒@24fps）实测采样峰值 8.52G 通过；**>121 帧模型会"回弹"首帧**（社区反馈，未深测）。

---

## 四、踩坑大全（全部真实踩过）

### 致命级（不处理直接炸，见顶部警告框）

1. **VAEDecodeTiled 默认参数必炸**：512/64/64/8 时单块（512px×64帧 fp32）激活 6.33GB → OOM。唯一可靠 `128/32/8/4`（25 组"分辨率×帧数"矩阵全过，峰值 2.7-3.4G）。
2. **untiled 解码崩进程**：`VAEDecode` 触发 ROCm 硬崩溃（GPU core dump），不是普通 OOM。
3. **ROCm 显存碎片污染**：连续 OOM 尝试会让后续必过组合假 OOM（720P@81帧保底参数在 4 次 OOM 后炸、全新状态 2.73G 轻松过）。**遇 OOM 先重启再降参数**。
4. **fp32 VAE 走低显存加载路径有非确定性**：同参数偶发 OOM（冷启动首测 8.31G 炸、重启后 6.27G 过）。
5. **🔴 ComfyUI-Manager 会把 torch 换成 CUDA 版，直接毁掉 ROCm 环境**（本机真实发生，环境级灾难）：Manager 的依赖解析不区分 ROCm/CUDA，装/升级节点时可能把 `torch 2.5.1+rocm6.2` 拉成 CUDA 版（连带几十个 `nvidia-*`/`cuda-*` 包）。**本机残留证据**：`pip3 list` 里至今仍有 33 个 NVIDIA/CUDA 包（9月6日装入，`nvidia-cublas-cu12`、`cuda-toolkit 13.0.3` 等），9月25日靠 `pip install --force-reinstall torch-2.5.1+rocm6.2-*.whl` 才救回（日志见 `~/torch_install.log`）。
   **已做的封堵**（`ComfyUI/user/__manager/config.ini`）：
   ```ini
   downgrade_blacklist = torch, torchvision, torchaudio, transformers, safetensors, comfy_kitchen
   allow_pip_install = False
   ```
   **防御建议**：① 保持上面两项配置；② 用 Manager 装节点前先 `cp -r ~/.local/lib/python3.10/site-packages/torch* /tmp/torch_backup/`；③ 事后**必查** `python3 -c "import torch;print(torch.version.hip)"` 是否为 `6.2.x`（出现 `None` 就是被换成 CUDA 版了）；④ 真被换了就按上文 `--force-reinstall` 重装 rocm wheel（需本地留好 whl）。

### 性能/画质级

6. **`--fp32-vae` 的隐性代价**：VAE 权重 fp32 占 2.7GB（bf16 仅 1.35GB），且 ComfyUI 显存预估公式（~10.8GB）强制 VAE 进 lowvram 部分加载——这是默认 tile 参数在别处能用、这里炸的根因之一。
7. **bf16 VAE 是权衡不是免费午餐**：输出与 fp32 逐像素一致、激活减半（256px/时序32 可过，4.92G），**但 gfx1031 的 MIOpen conv 路径下 bf16 解码慢 60%**（480P@81帧：199s→321s）。fp32+小 tile 仍是默认推荐。
8. **512px tile 在 fp32/bf16 下都 OOM**；256px 是 bf16 甜点位（反超 fp32 速度）。
9. **18 步有彩色斑块**（高频区域如草地），35 步干净；步数不影响显存只影响时间。
10. **tiledvaelite (LTTiledVAEDecode) 输给官方**：同规模 192.7s/7.26G vs 184.6s/4.92G。
11. **hipBLASLt 警告可忽略**：`Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas`——gfx1031 伪装 gfx1030 的正常回退。
12. **🔴 GGUF 底模挂 LoRA 会静默打折（本机真实踩过）**：GGUF 是量化权重，LoRA 走 `comfy.lora.calculate_weight` 对权重做 patch——在 Q4_K_M 这类量化权重上效果会衰减，且**不报错、不提示**，你只会觉得"这 LoRA 好像没生效/变弱了"。本机实测环境里 `ComfyUI-GGUF` 的 `nodes.py` 只提供 `UnetLoaderGGUF`/`CLIPLoaderGGUF` 系列（无内建 LoRA 合并），Wan2.2 又是 GGUF 底模，两者直接组合会踩。
    **规避**：① 用 safetensors 非量化底模挂 LoRA（最稳）；② 或在加载时先合并 LoRA 再量化（离线 merge）；③ 或换 `WanVideoWrapper` 等支持 GGUF+LoRA 的节点（未在本机实测，不做效果承诺）。

### 测试方法级（开源脚本已全部修掉）

13. **裸 SDPA ≠ ComfyUI 实际行为**：独立 benchmark 里 `F.scaled_dot_product_attention` 在 8K token（batch1）就 OOM（math 后端物化 12·N²·4B 的 fp32 注意力矩阵），但 ComfyUI 实测 75K token 只占 ~8.5G——**它对 query 做了分块**。别拿裸 torch 的结论吓退自己。
14. **0.5s HTTP 轮询漏峰**：`/system_stats` 0.5s 采样把峰值读低 ~1.5GB；改 0.1s 直读 sysfs `mem_info_vram_used` 才是真峰值（`scripts/driver.py` 已实现，用法见第六节）。
15. **ComfyUI 节点缓存会伪造"20 倍提速"**：重复提交完全相同的图（同参数同种子）命中缓存秒回——曾据此误判 bf16 提速 20×，实为缓存命中。**测速必须换种子击穿缓存**。
16. **端口抢占**：kill 旧 ComfyUI 不彻底时，新进程 `Port 8188 already in use` 静默失败，测试全跑在旧配置上——本仓库测试就栽过一次（整轮"bf16 数据"实为 fp32）。**切换配置后必须验证 `ss -ltn` 归属 + 进程 cmdline**。
17. **`pgrep -f "main.py"` 会匹配到执行它的 shell 自身**（自杀）；脚本里用 `pgrep -f "main[.]py"` 括号技巧规避。
18. **awk 字符串比较把显存峰值读小一个数量级**（SDXL 测试新踩）：`awk '{if($2>m)m=$2}'` 对 `7406MB` 这类带单位字段走**字典序**，`"904MB" > "7406MB"`（'9'>'7'），整轮峰值被误报成 904MB——比空闲占用还低才发现不对。改用 Python 数值比较立刻正常。**shell 统计带单位字段先 `gsub` 剥掉非数字**。
19. **`status=success` 会骗人——黑图静默失败**（SDXL 测试新踩）：1216×1824 bucket 采样 86s 正常跑完、无报错无 NaN 日志，输出却是**纯黑图**（mean=0, std=0），换种子/换采样器/tiled 解码全部复现。**任何生成测试驱动必须带亮度校验**（`mean<3` 判黑）——本仓库 Wan 版 `scripts/driver.py` 早有此检查，SDXL 新驱动初版没带，差点把黑图记成"通过"。

---

## 五、已知不可用清单（本机实测/确认，省你试错时间）

| 项目 | 状态 | 原因 |
|---|---|---|
| `torch.compile` | ❌ 不可用 | triton 3.8.0 与 torch 2.5.1 版本不匹配；升级 triton 又与 ROCm wheel 生态冲突 |
| xformers / SageAttention / FlashAttention | ❌ 不可用 | RDNA2（gfx1031）无官方支持，SDPA 落 math 后端（靠 ComfyUI 内部分块救回） |
| `VAEDecode`（untiled，**仅指 Wan 视频 VAE**） | ❌ 禁用 | 触发 ROCm 硬崩溃（见踩坑 2）；SDXL 自带图片 VAE 的 untiled 解码不受此限，实测正常（docs/sdxl-novaanimexl-test.md） |
| VAEDecodeTiled 默认参数 | ❌ 禁用 | 必 OOM（见踩坑 1） |
| tiledvaelite LTTiledVAEDecode | ❌ 不推荐 | 比官方慢且费显存（见踩坑 10） |
| WanVideoWrapper 直接加载 GGUF | ❌ 不支持 | 需另下其自有格式模型 |
| SeedVR2 视频超分 | ❌ 排除 | 社区共识需 12GB+ 显存 |
| comfy-kitchen triton 后端 | ⚠️ 自动禁用 | ImportError 回退，正常 |
| SDXL `1216×1824` / `1824×1216` bucket | ❌ 必出纯黑 | 形状级静默失败（见踩坑 19），SDXL 官方标准桶之一；同比例改用 `1152×1728`（实测正常） |
| GGUF 底模直接挂 LoRA | ⚠️ 效果打折且无提示 | 量化权重上 patch LoRA 会衰减（见踩坑 12）；改用 safetensors 底模或先 merge |
| ComfyUI-Manager 自动装/升级 torch | ❌ 严禁 | 会把 ROCm 版换成 CUDA 版（见踩坑 5）；已用 `downgrade_blacklist` + `allow_pip_install=False` 封堵 |
| `hipBLASLt` 警告 | ⚠️ 可忽略 | gfx1031 伪装 gfx1030 的正常回退（见踩坑 11），不用管 |

---

## 六、耗时曲线与复现

### 耗时曲线（18 步，实测锚点 ◆）

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

### 📹 输出格式说明（**跑完找不到 mp4 不是失败**）

本仓库工作流最后接的是 **`SaveAnimatedWEBP`**，输出是 **`.webp` 动图**（默认在 `ComfyUI/output/`），**不是 mp4**：

- 为什么用 webp：本机**没有 ffmpeg 二进制**，`SaveAnimatedWEBP` 由 ComfyUI 原生写出，无需编码器依赖。
- 想要 mp4：① 换成 `SaveWEBM` 节点（ComfyUI 内置，需系统有 ffmpeg）；② 或用已装的 **PyAV**（`av 17.1.0`）把 webp/帧序列转封装成 mp4（本机替代 ffmpeg 的既定方案）；③ 或直接把 webp 丢进剪辑软件。
- 帧率：工作流默认 `16 fps`（33 帧 ≈ 2 秒播放）；链式 I2V 段按 `24 fps` 生成再拼接。
- **不要**因为「output 里没有 .mp4」就判定生成失败——去看有没有 `.webp`。

### 显存监控脚本用法（`scripts/driver.py`）

```bash
# 依赖: 运行中的 ComfyUI + python3(numpy/pillow)
python3 scripts/driver.py scripts/phase1_queue.jsonl
# 输出追加到 results.jsonl，每行:
# {id, status: ok/oom/error/timeout, wall_sec, peak_vram_gb,
#  video:{frames, mean_brightness, black, nan}}
# 峰值 = 0.1s 间隔直读 /sys/class/drm/card1/device/mem_info_vram_used 的最大值
# 黑屏判定: 全帧亮度均值 < 3; NaN 判定: 任一帧含 NaN 像素
```

**队列文件（`.jsonl`）格式**——每行一个测试用例，自己写也很简单（生成器见 `scripts/make_queue.py`）：

```json
{"id": "480p_33f_18s", "width": 832, "height": 480, "frames": 33, "steps": 18, "cfg": 5.0, "sampler": "dpmpp_2m_sde_heun", "scheduler": "sgm_uniform", "seed": 12345, "prompt": "a cat walking in a garden, sunlight", "negative": "lowres, blurry"}
```

| 字段 | 含义 | 备注 |
|---|---|---|
| `id` | 用例名 | 进 results.jsonl 用于对照 |
| `width`/`height` | 分辨率 | 需为 16 的倍数 |
| `frames` | 帧数 | **必须 4n+1**（33/49/81/121…） |
| `steps`/`cfg`/`sampler`/`scheduler` | 采样参数 | 默认档见 TL;DR |
| `seed` | 种子 | **测速时必须每次不同**（击穿节点缓存，见踩坑 15） |
| `prompt`/`negative` | 提示词 | 一起喂给两个 CLIPTextEncode |

### 🔧 失败排查决策表（先查这里，再翻 19 条踩坑）

| 症状 | 最可能原因 | 第一步动作 |
|---|---|---|
| OOM（显存不足） | ROCm 碎片污染 > 参数真不够 | **先重启 ComfyUI** 用同参数再试一次；还炸再降 `temporal_size`→分辨率 |
| 进程直接消失/无报错 | untiled `VAEDecode` 触发 ROCm 硬崩溃 | 换 `VAEDecodeTiled` 128/32/8/4（见踩坑 2） |
| 输出纯黑图 | 形状级静默失败（如 SDXL 1216×1824） | 换邻近 bucket（1152×1728）；**别信 `status=success`**，先验亮度（见踩坑 19） |
| `torch.version.hip` 是 `None` | torch 被换成了 CUDA/CPU 版 | `--force-reinstall` 重装 rocm wheel（见踩坑 5） |
| 莫名跑得很慢/参数没生效 | 测试跑在旧 ComfyUI 进程上 | 查 `ss -ltn \| grep 8188` 归属 + 进程 cmdline（见踩坑 16） |
| LoRA 好像没效果 | GGUF 量化底模上 LoRA 静默打折 | 换 safetensors 底模或离线 merge（见踩坑 12） |

### 目录结构

```
├── README.md            本文件
├── docs/
│   ├── SUMMARY.md       完整测试报告（数据表）
│   ├── sdxl-novaanimexl-test.md  SDXL 图片生成实测（2026-10-07 新增）
│   └── deepseek-discussion.md  与 DeepSeek 三轮协作分析纪要
├── workflows/           可直接导入的工作流 JSON ×2
├── scripts/             driver.py + make_queue.py + SDXL 测试驱动/探针 ×4 + 全部测试队列 jsonl（原始结果只在 data/）
├── data/                原始测试结果 jsonl（~140 条，含 sdxl_results.jsonl 20 条）
└── frames/              证据帧：花斑对比、链式衔接验证、SDXL 成品图 ×5（含黑图证据）
```

## 声明

- 仓库内示例帧/结果为 AI 生成的普通素材（猫、动漫少女风景图），不含敏感内容
- 模型权重因体积/版权不随仓库分发，仅列清单与获取指引
- 数据为**单卡（RX 6750 GRE 10G，gfx1031）+ 上述软件栈**实测；不同驱动/内核/ROCm/MIOpen 版本组合可能有 ±10% 波动，个别结论（如 bf16 解码慢 60%、18 步花斑）是特定版本下的现象，换版本可能反转
- 大多数配置只跑 1-3 次，**"通过"指当次未失败，不代表长期稳定**；要长期挂机请自行复测
- 遇到问题先查"失败排查决策表"，再翻"踩坑大全"与"已知不可用"
