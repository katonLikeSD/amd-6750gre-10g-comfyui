# AMD 6750 GRE 10G 显存本地生图/生视频流程跑通参考

> 在 **RX 6750 GRE 10G（gfx1031 / RDNA2 / ROCm）** 上，用 ComfyUI 跑通 **Wan2.2 TI2V-5B 视频生成**（含图片生成环境）的完整实测数据与踩坑记录。所有数字为本地实测，非二手转述。

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

## 硬件与环境

| 项 | 值 |
|---|---|
| GPU | AMD RX 6750 GRE **10GB**（Navi 22, gfx1031，需 `HSA_OVERRIDE_GFX_VERSION=10.3.0`） |
| CPU / 内存 | Ryzen 5 7500F (6C12T) / 32GB |
| 软件栈 | Ubuntu 22.04 · PyTorch 2.5.1+rocm6.2 · ComfyUI 0.37.2 + ComfyUI-GGUF |
| 模型 | Wan2.2-TI2V-5B **Q4_K_M GGUF**（3.2GB）· umt5_xxl_fp8 · wan2.2_vae |
| 注意 | RDNA2 **无 flash/mem-efficient attention**，SDPA 走 math 后端（见下文"意外结论"） |

模型获取（不随仓库分发）：Wan2.2 TI2V-5B 权重 Apache-2.0，GGUF 量化版在 HuggingFace 搜 `Wan2.2-TI2V-5B-Q4_K_M-GGUF`；umt5 / wan2.2_vae 为 ComfyUI 官方模板同款。

## 关键实测结论

1. **采样显存几乎不随帧数/分辨率增长**——ComfyUI 的注意力对 query 自动分块。裸 PyTorch SDPA 在 8K token 就 OOM，但 ComfyUI 实测 1280×720@81帧（75K token）采样峰值也只有 ~8.5GB。**10G 卡能跑满 Wan2.2 5B 的全部官方分辨率档位**。
2. **VAE 解码才是雷区**：解码显存 ∝ 每块（像素面积 × 时序帧数），fp32 下默认参数单块就要 6.33GB。`128/32/8/4` 是 25 组"分辨率×帧数"矩阵里唯一全过的参数。
3. **untiled 解码（VAEDecode 不切块）会直接把 ComfyUI 进程崩掉**——ROCm 硬 Abort（`GPU core dump failed`），不是普通 OOM。永远别在这张卡上试。
4. **121 帧直接生成可行**（峰值 8.52GB，5秒@24fps）；社区反馈 >121 帧模型会"回弹"首帧。
5. **尾帧链式 I2V 可行**：用上一段尾帧作 `start_image`，核心节点路径衔接自然、尾部无噪声（kijai wrapper 的"末4帧噪声"坑在核心节点不存在）。每段 4.2 分钟。
6. **`--fp32-vae` vs `--bf16-vae` 权衡**：bf16 输出与 fp32 逐像素一致、激活减半（256px 大 tile 可过），但 gfx1031 的 MIOpen 卷积路径下 bf16 解码**慢 60%**。默认推荐 fp32。
7. **画质**：18 步在高频区域（草地）会有彩色斑块，35 步干净；步数不影响显存只影响时间。
8. **ROCm 显存碎片**：连续 OOM 尝试会污染后续测试（必过的参数也会假 OOM），遇到先重启 ComfyUI 再测。

## 耗时曲线（18 步，实测锚点 ◆）

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

公式：`T ≈ 34s(文本编码) + 步数×步时 + 帧数×2.46s(480P解码)`；步时 ∝ token 数（480P@33帧 ≈ 8.6s，480P@81帧 ≈ 27.5s）。

## 目录结构

```
├── README.md            本文件
├── docs/
│   ├── SUMMARY.md       完整测试报告（全部数据表）
│   └── deepseek-discussion.md  与 DeepSeek 的三轮协作分析纪要
├── workflows/           可直接拖入 ComfyUI 的工作流
│   ├── wan22_t2v_832x480_33f.json        日常最优档（文生视频）
│   └── wan22_i2v_chain_segment.json      尾帧链式延长段
├── scripts/             测试基础设施
│   ├── driver.py        API 队列驱动（0.1s sysfs 显存峰值监控 + 黑屏/NaN 检测）
│   ├── make_queue.py    测试队列生成器
│   └── *.jsonl          全部测试用例定义
├── data/                原始测试结果（~60 条记录，含峰值显存/耗时/像素统计）
└── frames/              证据帧：花斑对比、链式衔接验证
```

## 如何复现

```bash
# 1. 启动 ComfyUI（见 TL;DR）
# 2. 拖入 workflows/wan22_t2v_832x480_33f.json
# 3. 跑一条 ≈ 4 分钟。改 length=81 → 11.6 分钟，改 1280×704 → 720P 档
# 4. 延长：跑完一段后，把输出尾帧存为 input/chain_seed.png，
#    导入 wan22_i2v_chain_segment.json 生成下一段，PyAV/剪辑软件拼接
# 5. 批量测试：python3 scripts/driver.py scripts/phase1_queue.jsonl
```

## 声明

- 仓库内示例帧/结果为 AI 生成的普通素材（猫），不含任何敏感内容
- 模型权重因体积/版权不随仓库分发，仅提供获取指引
- 数据为单卡单次实测，不同驱动/内核版本可能有 ±10% 波动
