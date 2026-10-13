# 闪烁（时域抖动）对策实测

> 2026-10-12 实测。起因：用户肉眼反馈"所有版本闪烁都比较严重"，为当前第一待解问题。
> 测试条件固定：spmix + Turbo LoRA 1.0 / 4 步 / CFG 1 / euler+simple / seed 42 / **33 帧 @1280×704** /
> 情趣内衣+房间**固定镜头静止场景**（提示词见 `docs/mixer-lingerie-static-test.md`）。
> 指标：**帧间差**（相邻帧逐像素绝对差均值，中位/max）与**锐度**（Laplacian 方差，取中间帧）。

## 一、ModelSamplingSD3 的 shift 值扫描

| 配置 | 耗时 | 帧间差 中位 | 帧间差 max | 锐度 |
|---|---|---|---|---|
| 对照（不加 SD3 节点） | 276 s | 6.74 | 9.56 | 71 |
| **shift = 3** | 238 s | **5.63（−16%）** | **7.47（−22%）** | 56（−21%） |
| shift = 5（官方模板默认） | 238 s | 6.44 | 8.75 | 68 |
| shift = 7 | 238 s | 6.70 | 9.37 | 68 |

**结论**：shift 越低越稳、但细节越软；**shift=3 抖动最低**；shift=7 最抖。不加节点 ≈ shift 5–7。
→ **日常建议 shift 3–5**（要稳取 3，要锐取 5）。

## 二、时域平滑（TemporalSmoothingLite）A/B

同一次运行同时输出"原始"与"平滑"两版，隔离平滑本身的收益（原始列两行相同 = 同一批帧）：

| 配置 | 帧间差 中位 | 帧间差 max | 锐度 |
|---|---|---|---|
| 原始（未平滑） | 5.63 | 7.47 | 56 |
| **k5 / σ1.0 / strength 1.0** | **3.04（−46%）** | **4.20（−44%）** | 26（−54%） |
| k3 / σ1.0 / strength 1.0 | 3.58（−36%） | 4.76（−36%） | 30（−46%） |

### strength 平衡点扫描（k5 / σ1.0 / shift3，同 seed 同场景）

| strength | 帧间差中位 | 帧间差 max | 锐度 | 相对无平滑 |
|---|---|---|---|---|
| 0（无平滑） | 5.63 | 7.47 | 56 | — |
| **0.5（推荐）** | **4.29** | **5.68** | **37** | 抖动 −24% / 锐度 −34% |
| 0.7 | 3.77 | 5.03 | 32 | 抖动 −33% / 锐度 −43% |
| 1.0 | 3.04 | 4.20 | 26 | 抖动 −46% / 锐度 −54% |

**★ 双赢组合（实测）**：挂 LoRA 时**抖动本质是高频时域噪声**，而"不挂 LoRA"只是用"欠去噪的软"换低抖动。
对比：不挂 LoRA = 抖动 4.81 / 锐度 27；**挂 LoRA + shift3 + 平滑 strength 0.5 = 抖动 4.29 / 锐度 37**
→ **两个指标都更好**。推荐日常配置：**LoRA 1.0 + shift 3 + TemporalSmoothingLite(k5, σ1.0, strength 0.5)**。

**结论**：时域平滑能砍掉 24–46% 的抖动量，代价是锐度（比例约 1 : 1.3~1.4，属硬权衡）。
闪烁本质是"逐帧高频噪声 + 微抖动"——时间平均在压噪声的同时也抹掉静态细节，属硬性权衡。
→ **下一步（下次测）**：`strength` 取 **0.3 / 0.5 / 0.7** 找平衡点（节点已支持该参数）；
备选：`mode=optical_flow`（运动感知，代价更高）、或先降 shift=3 再轻平滑。

## 三、已交付的工具

`custom_nodes/temporal_smoothing_lite/`（本仓库自带，拷贝到 `ComfyUI/custom_nodes/` 重启即用）：

- 节点名 **TemporalSmoothingLite**（显示名"时域平滑（轻量）"）
- 参数：`kernel_size`（时间窗，奇数，默认 5）/ `sigma`（默认 1.0）/ `strength`（0=不处理，1=全用平滑结果）
- 接线：**VAE Decode → TemporalSmoothingLite → 保存视频**
- 零外部依赖（仅 torch）；同一算法来自 `siraxe/ComfyUI-SA-Nodes-QQ` 的 TemporalSmoothingNode——
  该包因缺 `omegaconf` 等依赖**整包导入失败**（所以机器上虽装有此包但节点一直不可见），故独立重写为单文件节点。

## 四、复现

```bash
python3 scripts/scan_flicker_shift.py        # shift 3/5/7 + 对照（4 用例，约 18 分钟）
python3 scripts/scan_flicker_smooth.py 3 5 1.0   # shift=3 下 k5/σ1 的 raw vs sm A/B
```
（脚本在 `~/video10s/`；结果 JSON：`flicker_shift.json`、`flicker_smooth.json`、`flicker_smooth_k5.json`）