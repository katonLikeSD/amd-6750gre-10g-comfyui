# 固定镜头 · 静止展示 · 场景更换测试（spmix ± Turbo LoRA @720p）

> 2026-10-10/11 实测。目的：验证"**固定镜头 / 禁止人物大幅动作 / 镜头聚焦 / 仅展示身材**"这一拍摄约束在
> Wan2.2 5B（spmix 社区混剪版）上的可行参数；并测试同一约束下**更换场景**（泳池 → 房间+情趣内衣）是否需要改参数。
> 所有数字为本机实机墙钟耗时（含加载/编码/采样/解码），seed 固定 42 便于逐像素对比。

## 一、提示词写法（核心：把约束写死，并在负向里封杀运镜/大幅动作）

**正向（模板，`{}` 内为可替换场景）**

```
A static locked-off camera shot, absolutely no camera movement. A beautiful adult
woman wearing {Wardrobe} stands still in {Scene}, holding a calm pose and quietly
showing her figure. The camera is focused on her body; the frame is centered on her
torso and hips. Only very subtle motion: slow natural breathing and a faint breeze
moving a few strands of her hair. No walking, no turning, no arm gestures, no large
body movement. {Light} , soft shadows, clean uncluttered background, slightly blurred
background, sharp focus on the subject, high quality, detailed, photorealistic.
```

**负向（本约束专用，必须带上）**

```
camera movement, pan, zoom, dolly, tracking shot, walking, running, turning around,
jumping, dancing, waving arms, large body motion, motion blur, blurry, low quality,
deformed, distorted face, bad hands, extra fingers, extra limbs, flickering,
watermark, text, lens flare, bloom, overexposed, blown highlights, glow, hazy
```

**场景替换实测两例**（只改 `{Wardrobe}` / `{Scene}` / `{Light}`，其余一字不动，参数完全不变——**换场景不需要改参数**）：

| 例 | Wardrobe | Scene | Light |
|---|---|---|---|
| A 泳池 | `elegant black lace lingerie` → `a white bikini` | `beside a sunlit swimming pool` | `Soft warm sunlight, water reflections` |
| B 房间 | `elegant black lace lingerie` | `in a softly lit bedroom` | `Warm indoor lamp light` |

## 二、参数（两例通用）

| 项 | 值 |
|---|---|
| 模型 | `wan225bi2vspmixver021n_v21.gguf`（社区混剪版，非官方） |
| LoRA | `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16`（强度 **1.0**；见结论①） |
| 分辨率 | **1280×704**（720p） |
| 步数 / CFG / 采样器 | **4 步 / CFG 1.0 / euler / simple**（蒸馏档配套参数） |
| VAEDecodeTiled | 128 / 32 / 8 / 4（固定不变） |
| seed | 42（fixed） |
| 帧数 | 33（快检）→ 121（5 秒 @24fps） |

## 三、结果

### 例 B（房间+情趣内衣，本约束）

| 配置 | 33 帧 | 121 帧（5 秒） | 锐度（Laplacian 均值） | 帧间差（中位） |
|---|---|---|---|---|
| 无 LoRA | 260 s / 9.31 G | 4390 s\* / 9.65 G | 13–27（**软/欠去噪**） | 4.81 |
| **Turbo LoRA 1.0** | 244 s / 9.24 G | **951 s** / 9.50 G | **119–172（锐利）** | 7.54 |

\* 4390 s 为异常值（同模型第二次运行仅 951 s），疑为 MIOpen 首次编译 121 帧 720p 解码核 + 抢占；未完全查明，如实标注。

### 例 A（泳池，同约束）扫描

| 配置 | 33 帧耗时 | 峰值显存 |
|---|---|---|
| 4 步 无 LoRA | 280 s | — |
| 4 步 LoRA 1.0 | 245 s | — |
| 20 步 无 LoRA | 687 s | — |
| 20 步 LoRA 1.0 | 709 s | — |

## 四、结论

1. **走 4 步必须挂 Turbo LoRA**：同 seed/同提示词/同场景下，挂 LoRA 的锐度是不挂的 **5–10 倍**（119–172 vs 13–27）。
   spmix 本身没有蒸馏，4 步裸跑只是"欠去噪的软图"；LoRA 提供蒸馏轨迹后才成立。
2. **5 秒（121 帧）可行**：挂 LoRA 版 951 s（16 分钟）@720p，画面稳定、构图不变。
3. **换场景不需要改参数**：只替换提示词里的 `{Wardrobe}/{Scene}/{Light}` 即可（泳池↔房间已验证）。
4. **画面风格差异（肉眼与用户判断一致）**：无 LoRA 版更"真实/自然"（软），LoRA 版更锐利但更"AI 光泽"、帧间差更大。
5. **未解问题：闪烁**（用户实测反馈"所有版本闪烁都比较严重"）。量化：本约束静态场景下帧间差中位 4.8–7.5。
   候选对策（按性价比）：① 用无 LoRA 版（帧间差更低）；② 输出后做时域平滑/补帧（RIFE 类）；③ 降分辨率换更多步数；
   ④ 一次生成更长片段（长片段的时域一致性更好）；⑤ 后期做帧间一致性约束（需第三方节点）。
6. **构图注意**：提示词里 `frame is centered on her torso and hips` 会让**头部出画**（实测如此）。
   要保留脸，把该句改成 `the frame shows her full body` 或 `head and torso in frame`。

## 复现

```bash
python3 scripts/scan_lingerie_room.py 33    # 快检
python3 scripts/scan_lingerie_room.py 121   # 5 秒
```
（脚本在 `~/video10s/`，参数化见文件顶部；工作流见 `workflows/wan22_10s_turbo4step.json`，已内置 I2V+LoRA 模块）