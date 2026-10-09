#!/usr/bin/env python3
"""跨段色彩配平（饱和度平滑校正）：把长片后半段的饱和度漂移平滑拉回段首基线。

背景：链式续写段会渐进偏饱和（实测 0.70 → 0.90+），接缝像素差虽小，但整体观感会"越来越艳"。
做法：HSV 空间对 S 通道乘以逐帧增益 g_i = target / s_i，再做时间平滑（移动平均），避免闪变。
用法：python3 grade_match.py <in.webp> <out_prefix>
"""
import os, sys
import numpy as np
from PIL import Image
import cv2

BASE_FRAMES = 60      # 取前 N 帧的饱和度均值作为目标基线
SMOOTH = 48           # 增益曲线时间平滑窗口（帧）


def sat_of(a):
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def main(src, out_prefix):
    im = Image.open(src)
    n = im.n_frames
    frames = []
    for i in range(n):
        im.seek(i)
        frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    frames = np.stack(frames)
    print("frames:", n, "sat mean:", round(float(np.mean([sat_of(f) for f in frames])), 3))

    # 逐帧 HSV 的 S 均值
    s_means = []
    hsvs = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsvs.append(hsv)
        s_means.append(float(hsv[:, :, 1].mean()) / 255.0)
    s_means = np.array(s_means)
    target = float(s_means[:BASE_FRAMES].mean())
    gain = np.clip(target / np.maximum(s_means, 1e-6), 0.75, 1.15)

    # 时间平滑（edge-padded 移动平均）
    k = SMOOTH
    pad = np.pad(gain, (k // 2, k - 1 - k // 2), mode="edge")
    kernel = np.ones(k) / k
    gain_s = np.convolve(pad, kernel, mode="valid")

    out = []
    for i, hsv in enumerate(hsvs):
        h = hsv.copy()
        h[:, :, 1] = np.clip(h[:, :, 1] * gain_s[i], 0, 255)
        rgb = cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB)
        out.append(rgb)
    print("target sat:", round(target, 3), "| gain range:", round(float(gain_s.min()), 3),
          round(float(gain_s.max()), 3))
    print("after sat mean:", round(float(np.mean([sat_of(f) for f in out])), 3),
          "| first60:", round(float(np.mean([sat_of(f) for f in out[:BASE_FRAMES]])), 3),
          "| last60:", round(float(np.mean([sat_of(f) for f in out[-60:]])), 3))

    webp_out = out_prefix + "_graded.webp"
    imgs = [Image.fromarray(f) for f in out]
    imgs[0].save(webp_out, save_all=True, append_images=imgs[1:], duration=41, loop=0,
                 lossless=True, quality=100)

    import av
    from fractions import Fraction
    mp4_out = out_prefix + "_graded.mp4"
    c = av.open(mp4_out, mode="w")
    st = c.add_stream("libx264", rate=Fraction(24, 1))
    st.width, st.height = out[0].shape[1], out[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.options = {"crf": "16", "preset": "medium"}
    for i, f in enumerate(out):
        vf = av.VideoFrame.from_ndarray(f, format="rgb24")
        vf.pts = i
        vf.time_base = Fraction(1, 24)
        for pkt in st.encode(vf):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    print("written:", webp_out, mp4_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])