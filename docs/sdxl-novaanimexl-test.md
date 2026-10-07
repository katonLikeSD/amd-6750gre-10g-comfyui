# SDXL 图片生成实测 — novaAnimeXL_ilV190（2026-10-07）

> 背景：README 模型清单里"SDXL 系动漫 checkpoint 缺失"一项已于 2026-10-07 补回并实测通过。
> 本篇记录完整数据，替换 README 中原先"基于 SDXL 经验"的图片侧预测。

## 测试对象

| 项 | 值 |
|---|---|
| 模型 | `novaAnimeXL_ilV190.safetensors`（6.94 GB，SDXL 动漫 checkpoint，自带双 CLIP + VAE） |
| 放置目录 | `models/checkpoints/` |
| 环境 | 与视频测试同一套：ComfyUI 0.37.2 + torch 2.5.1+rocm6.2 + `HSA_OVERRIDE_GFX_VERSION=10.3.0`，启动带 `--fp32-vae` |
| 显存监控 | 0.1s 直读 sysfs `mem_info_vram_used`（方法见踩坑 12） |

## 结果总览

| # | 分辨率 | 步数 | 种子 | 端到端耗时 | 状态 |
|---|---|---|---|---|---|
| run1 | 1024×1024 | 30 | 20261007 | **40.1s**（含冷加载模型） | ✅ success |
| run2 | 832×1216（竖构图） | 25 | 20261008 | **32.0s** | ✅ success |
| run3 | 1024×1024 | 1 | 20261009 | 4.0s | ✅（固定开销探针，输出为噪声属正常） |
| run4 | 1024×1024 | 15 | 20261011 | 23.1s | ✅（显存核验轮） |

采样器统一 `dpmpp_2m_sde` + `karras`，cfg 7.0。

### 速度

- 固定开销（CLIP 编码 + VAE 解码 + 轮询）≈ **4.0s**（run3 探针）。
- 稳态步速 ≈ **1.2s/it**（1MP 级）：run1 (40.1−4)/29 ≈ 1.24；run2 (32−4)/24 ≈ 1.17；run4 (23.1−4)/14 ≈ 1.36（run4 有采样线程抢 CPU，略偏高）。
- 日常参考：**1024×1024 × 30 步 ≈ 36s/张**（模型常驻时）。

### 显存

- 采样峰值 **7286–7406 MB / 10224 MB**（run4 精确核验 7286MB；run1 时段采样器记录 7406MB）。
- 服务器日志构成：UNET `full load 4897MB` + CLIP 编码期 1561MB（编完即卸载）+ VAE 319MB + torch 上下文 ~0.4GB。
- ComfyUI 0.37.2 默认启用 async weight offloading / RAM pressure cache，但本卡余量足够，SDXL 全程全量驻留显存，无 lowvram 分块。
- 结论：**10G 卡跑 SDXL 1MP 级图片余量充足（约 2.8GB 空闲）**，无需任何显存优化参数。

### 与 Wan 视频 VAE 警告的边界（重要）

README 顶部两条 CAUTION（VAEDecodeTiled 默认参数 OOM、untiled `VAEDecode` 硬崩溃）**只针对 Wan 48 通道视频 VAE**。SDXL checkpoint 自带的图片 VAE 在 `--fp32-vae` 下用默认 `VAEDecode`（untiled）解码 1024² / 832×1216 完全正常，峰值仅多占约 0.3GB。**不要**给 SDXL 工作流套 tile 128/32/8/4 参数。

### 画质

- 30 步输出干净：无彩色斑块（对比 Wan 18 步"花电"现象）、无解剖错误、构图与提示词吻合。
- 证据图：`frames/sdxl_novaAnimeXL_1024x1024_s30.png`、`frames/sdxl_novaAnimeXL_832x1216_s25.png`。
- 1 步探针图（run3）为纯噪声，属正常，未收录。

## 推荐参数（直接抄）

| 项 | 值 |
|---|---|
| 分辨率 | 1024×1024（方图）或 832×1216（竖图，SDXL 原生桶） |
| KSampler | 30 步 · cfg 7.0 · `dpmpp_2m_sde` · `karras` |
| VAE 解码 | 默认 `VAEDecode`（untiled），**不需要** tiled |
| 预期 | ≈36s/张，峰值 ≈7.3GB |

## 复现

```bash
# 前提: ComfyUI 已按 README 启动命令运行
python3 scripts/sdxl_test_driver.py run1_1024 1024 1024 20261007 30 7.0 dpmpp_2m_sde karras
# 显存核验（采样线程 + 生成 + 峰值统计一体）:
python3 scripts/sdxl_verify_vram.py
# 原始数据: data/sdxl_results.jsonl
```

## 本篇新增踩坑

**awk 字符串比较把峰值读小一个数量级**：`awk '{if($2>m)m=$2}'` 处理 `7406MB` 这类带后缀字段时走**字典序**（"904MB" > "7406MB"，因为 '9' > '7'），导致整轮 run1-3 的峰值被误报为 904MB——比空闲占用还低。核验方式：同一脚本改用 Python 数值比较后峰值立刻正常（7286MB）。凡是用 shell 工具统计带单位字段，先 `gsub` 剥掉非数字再比较。
