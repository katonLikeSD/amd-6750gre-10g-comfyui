# SDXL 图片生成实测 — novaAnimeXL_ilV190（2026-10-07，含第二轮扩展矩阵）

> 背景：README 模型清单里"SDXL 系动漫 checkpoint 缺失"一项已于 2026-10-07 补回并实测通过。
> 本篇记录完整数据（两轮共 18 次生成），替换 README 中原先"基于 SDXL 经验"的图片侧预测。

## 测试对象

| 项 | 值 |
|---|---|
| 模型 | `novaAnimeXL_ilV190.safetensors`（6.94 GB，SDXL 动漫 checkpoint，自带双 CLIP + VAE） |
| 放置目录 | `models/checkpoints/` |
| 环境 | 与视频测试同一套：ComfyUI 0.37.2 + torch 2.5.1+rocm6.2 + `HSA_OVERRIDE_GFX_VERSION=10.3.0`，启动带 `--fp32-vae` |
| 显存监控 | 0.1s 直读 sysfs `mem_info_vram_used`（方法见踩坑 12） |

## 第一轮：基础验证

| # | 分辨率 | 步数 | 端到端耗时 | 状态 |
|---|---|---|---|---|
| run1 | 1024×1024 | 30 | 40.1s（含冷加载模型） | ✅ success |
| run2 | 832×1216（竖构图） | 25 | 32.0s | ✅ success |
| run3 | 1024×1024 | 1 | 4.0s | ✅（固定开销探针，输出为噪声属正常） |
| run4 | 1024×1024 | 15 | 23.1s | ✅（显存核验轮，峰值 7286MB） |

提示词统一（樱花白裙少女），采样器 `dpmpp_2m_sde` + `karras`，cfg 7.0。

## 第二轮：稳态速度矩阵（1024×1024，熟模型，全部换种子）

| # | 步数 | 采样器 | cfg | 耗时 | 结论 |
|---|---|---|---|---|---|
| A | 30 | dpmpp_2m_sde | 7 | 42.1s（重启后首轮含冷加载） | 稳态步速基准 |
| B | 20 | dpmpp_2m_sde | 7 | 26.0s | |
| C | 20 | dpmpp_2m | 7 | 26.0s | 非 SDE 不比 SDE 快 |
| D2 | 20 | euler_ancestral | 7 | 26.0s | 三采样器同速 |
| **E** | **15** | **dpmpp_2m** | 7 | **20.0s** | **20 秒档成立，画质干净** |
| F | 25 | dpmpp_2m | 7 | 32.0s | |
| G | 20 | dpmpp_2m | 5 | 26.0s | cfg 5 与 7 耗时相同 |

- **稳态步速 ≈ 1.15-1.17 s/it @1MP**，对采样器（dpmpp_2m / dpmpp_2m_sde / euler_ancestral）、cfg（5/7）均不敏感。
- 固定开销 ≈ 4.0s（run3 探针：CLIP 编码 + VAE 解码 + 轮询粒度）。
- 步数×1.17s + 4s 即可预估任意配置的端到端耗时。

## 高分辨率矩阵与显存上限

| # | 分辨率 | MP | 步数 | 解码方式 | 耗时 | 峰值显存 | 结果 |
|---|---|---|---|---|---|---|---|
| H | 1024×1536 | 1.57 | 25 | untiled | 56.2s | ≤8.4GB | ✅ 正常 |
| M | 1152×1728 | 1.99 | 25 | untiled | 74.1s | ≤8.4GB | ✅ 正常 |
| I | 1216×1824 | 2.22 | 25 | untiled | 86.1s | 8.4GB | ❌ **纯黑**（见下） |
| K | 1536×1536 | 2.36 | 25 | untiled | 90.1s | **9.15GB** | ✅ 正常 |
| J | 1024² ×batch2 | 2.10 | 20 | untiled | 50.1s | 8.15GB | ✅ 两张都正常 |

- 高分辨率步速近似线性于像素数：1.5MP ≈2.2s/it，2.0MP ≈2.9s/it，2.4MP ≈3.6s/it。
- **1536×1536 是本卡 untiled 实测上限附近**（峰值 9.15GB / 10GB）；batch2 @1024² 也安全（8.15GB）。
- 原先 README 的"1024×1536 需配合 tiled VAE"预测**证伪**——1.5MP 乃至 2.36MP untiled 都能过。

## ⚠️ 黑图 bug：1216×1824 / 1824×1216 必出纯黑（本机确定性复现）

**现象**：status=success、无报错、无 NaN 日志、耗时正常（86s），但输出 PNG 全黑（mean=0, std=0, 唯一色=1）。

**排查矩阵**（全部命中，无一幸免）：

| 变量 | 尝试 | 结果 |
|---|---|---|
| 种子 | 20261224 / 20261228 | 都黑 |
| 采样器 | dpmpp_2m / euler_ancestral | 都黑 |
| scheduler | karras / normal | 都黑 |
| VAE 解码 | untiled / **tiled 512** | 都黑 → **问题在采样阶段，非解码** |
| 方向 | 1216×1824 竖 / 1824×1216 横 | 都黑 |
| 邻近 bucket | 1024×1536、1152×1728、1536×1536 | 全部正常 |

**结论**：这是形状级（latent 152×228）的静默数值失败，与模型无关的通用结论未验证，但**在本机（gfx1031 + 本软件栈）该 bucket 不可用**。讽刺的是 1216×1824 是 SDXL 官方标准桶之一。

**规避**：用 1152×1728（同 2:3 比例，实测正常）或 1024×1536。**教训**：`status=success` ≠ 图能用——**任何批量测试必须做亮度校验**（mean<3 判黑，原仓库 `scripts/driver.py` 早有实现，本轮 SDXL 测试初版漏了，导致黑图差点被当成"通过"）。

## ⚠️ 偶发崩机事件（一次，未复现）

第一轮服务器连续跑了 4 次生成后，第 5 次（30 步）采样完成、`Requested to load AutoencoderKL` 瞬间触发 ROCm `Memory access fault → GPU core dump failed → Fatal Python error: Aborted`，整个进程死亡（SIGTERM 杀不掉、需 SIGKILL，进程状态 `ILl` 内存页被 GPU 锁死）。重启后同参数 + 连续 13 次生成（含 2.36MP）均未复现。符合踩坑 4"fp32 VAE 低显存加载路径非确定性"——**崩了就直接重启，别怀疑参数**。

## untiled VAEDecode 边界（修正版）

README 顶部两条 CAUTION（VAEDecodeTiled 默认参数 OOM、untiled 硬崩溃）**只针对 Wan 48 通道视频 VAE**。SDXL checkpoint 自带图片 VAE 在 `--fp32-vae` 下 untiled 解码实测到 1536×1536 正常（峰值 9.15GB），不需要 tiled。但注意两点：① 接近 10GB 上限，再往上（如 1664×2496）未测、不建议裸试；② 上面的黑图 bug 提醒——高分辨率跑完先验亮度再收工。

## 推荐参数（直接抄）

| 档位 | 配置 | 预期 |
|---|---|---|
| 快出图 | 1024×1024 · **15 步** · dpmpp_2m · karras · cfg 7 | **≈20s/张**，画质干净（实测 E） |
| 精修 | 1024×1024 · 30 步 · 同上 | ≈40s/张 |
| 竖构图 | 832×1216 或 1024×1536（**避开 1216×1824**） | 32s / 56s |
| 批量 | batch2 @1024² | 50s 两张，峰值 8.15GB |

## 复现

```bash
# 前提: ComfyUI 已按 README 启动命令运行
python3 scripts/sdxl_test_driver.py <name> <w> <h> <seed> <steps> <cfg> <sampler> <scheduler>
# 例: python3 scripts/sdxl_test_driver.py E 1024 1024 20261105 15 7.0 dpmpp_2m karras
# 高分辨率/黑图/批量核验（含 0.1s 显存采样 + 亮度校验）:
python3 scripts/sdxl_limit_probe.py
# 原始数据: data/sdxl_results.jsonl（18 条）
```

## 本篇新增踩坑

1. **awk 字符串比较把峰值读小一个数量级**：`awk '{if($2>m)m=$2}'` 处理 `7406MB` 这类带后缀字段时走**字典序**（"904MB" > "7406MB"，因为 '9' > '7'），导致整轮 run1-3 的峰值被误报为 904MB——比空闲占用还低。核验方式：同一脚本改用 Python 数值比较后峰值立刻正常（7286MB）。凡是用 shell 工具统计带单位字段，先 `gsub` 剥掉非数字再比较。
2. **黑图静默失败**：见上文 bug 章节。亮度校验（mean<3）必须进测试驱动，`status=success` 不可信。
