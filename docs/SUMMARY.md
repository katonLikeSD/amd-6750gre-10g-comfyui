# ComfyUI 极限测试完整报告（2026-09-27/28 夜间）

> **2026-09-28 00:52-01:00 补充实测（扩像素+延时方案验证）**
> - **121帧直接生成：✅ 通过**（1步探针 70.1s，采样峰值 **8.52G**）——A1 方案实证可行，121帧=5秒@24fps
> - **尾帧链式 I2V：✅ 通过**（核心节点 start_image 路径，18步/33帧 252.5s，峰值 **9.3G**）——第二段首帧与第一段尾帧衔接良好，**尾部无噪声**（DS 引用的 kijai wrapper 末4帧噪声坑在核心节点上不存在）
> - 链式测试帧：chain_f*.png（本目录）

> 硬件：RX 6750 GRE 10G（gfx1031, HSA_OVERRIDE_GFX_VERSION=10.3.0）· R5 7500F · 32G 内存
> ComfyUI 0.37.2 · torch 2.5.1+rocm6.2 · 启动参数 `main.py --fp32-vae`
> 与 DeepSeek 三轮协作分析的完整实测数据。原始数据：results*.jsonl（本目录）

## 一、最优工作流参数（已验证）

| 项 | 值 |
|---|---|
| 分辨率×帧数 | 832×480 @ 33帧（日常最优档） |
| KSampler | 18步 · cfg 5.0 · dpmpp_2m_sde_heun · sgm_uniform |
| VAEDecodeTiled | **tile_size=128 · overlap=32 · temporal_size=8 · temporal_overlap=4（必改，默认参数必OOM）** |
| 文本编码器 | umt5_xxl_fp8 挂 CPU（device=cpu；换 GPU 省 20s/新提示词，编码期峰值 9.1G） |
| 耗时/显存 | 244.5s/条 · 采样峰值 8.2G（0.1s sysfs 实测） |

耗时公式（18步）：T ≈ 34s(TE) + 18×步时 + 解码帧数×2.46s(480P)
- 480P：17帧≈2min，33帧≈4.1min(实测)，49帧≈7min，65帧≈9.6min，**81帧≈11.6min(实测)**
- 720P：33帧≈12min，81帧≈30min(推算)
- 35 步画质更好（花斑少），耗时×1.9

## 二、关键结论

1. **采样峰值几乎不随帧数/分辨率增长**（ComfyUI 注意力自动分块）：81帧 6.68G、**121帧 8.52G（实测通过）** → A1 直接长生成可行，上限约 121 帧（社区反馈超 121 帧会"回弹"首帧）
2. **VAE 解码才是雷区**：默认 tile 512/64/64/8 单块激活 6.33G 必 OOM；128/32/8/4 是唯一 25 组合全过参数；**untiled 解码会硬崩 ROCm（进程 Abort）绝对别用**
3. **花斑问题**：18步+832×480 出彩色斑块；35步 或 704×480 干净。步数对显存无影响只影响时间
4. **--bf16-vae**：输出与 fp32 逐像素一致、激活减半（256px/时序32 可过，4.92G），但 gfx1031 MIOpen 下解码慢 60%——权衡项非免费午餐
5. **tiledvaelite (LTTiledVAEDecode)** 输给官方 VAEDecodeTiled，不用
6. 512px tile fp32/bf16 都 OOM；720P@81帧 用 256/32/32/4 可过但 9.73G 压线（碎片污染下会假 OOM，重启后重测为准）
7. ROCm 碎片：连续 OOM 尝试会污染后续测试，出现"必过参数 OOM"先重启再测
8. 显存监控：0.1s sysfs（mem_info_vram_used）比 0.5s HTTP 轮询读数高 ~1.5G（能抓短峰）

## 三、下一步：扩像素+延长时间方案（与 DS 讨论定稿）

**A 延长时间**
- A1 直接 121 帧（5 秒@24fps）：✅ 本次已实测采样可行（8.52G）
- A2 尾帧链式 I2V 多工作流：核心节点 Wan22ImageToVideoLatent 的 start_image 路径已实测（chain_seg2）。注意 DS 引用的社区坑：kijai wrapper 的尾帧注入末 4 帧噪声 → 每段多生成 3-4 帧裁掉；进阶用 latent 空间拼接（VideoChunkTools 的 Blend Latent Chunks，需安装）
- A3 WanVideoWrapper context windows：不支持 GGUF，需重下模型，暂缓

**B 扩像素**
- B1 生成时直接 720P（1280×704 原生档）：已实测可行，12min/33帧
- B3 FlashVSR（节点已装在 WanVideoWrapper）：官方支持 Wan2.2 VAE，8G 档用 tiny-long + tiling + fp16；模型 ~3G 未下载；风险 BF16 张量不匹配（gfx1031 需最小验证后再下）
- B2 Real-ESRGAN 逐帧：保底，ROCm "buggy"（MIOpen 卷积坑同 bf16 VAE），单帧 <2G 显存
- ~~SeedVR2~~：需 12G+，排除

**C 补帧/合成**
- RIFE VFI（模型 12MB）16→32fps，用 PyAV（已装）写 20 行脚本替代 VHS；ffmpeg 二进制缺失，apt 安装可选
- LongLive（NVlabs，支持 Wan2.2 5B 长视频蒸馏）：NVIDIA 导向，ROCm 无文档，观察项

## 四、复现/继续

- 测试脚本：/tmp/comfy_test/{driver.py,make_queue.py,*.jsonl}（重启后 /tmp 清空，本目录有副本）
- 结果行格式：{id, status: ok/oom/error/timeout, wall_sec, peak_vram_gb, video:{frames,mean_brightness,black,nan}}
- 花斑对照帧：frame_p2full_*.png（本目录）
- 未完成：35步/704×480 对照被中止；FlashVSR 最小验证（待下载模型）；RIFE 脚本
- DS 对话在浏览器 DeepSeek 会话中（三轮记录）
