#!/usr/bin/env python3
"""最终成片：
(a) short10s_chain_241f_24fps.mp4    —— 链式方法原始输出（241帧@24fps，尾部有糊化，注释在文档）
(b) short10s_final_10s.mp4           —— 交付版：取已验证干净的前 210 帧 @21fps = 精确 10.000s，含饱和度配平
"""
import os
import numpy as np
from PIL import Image
import cv2

PROJ = os.path.expanduser("~/video10s")
CHECK = os.path.join(PROJ, "check")
OUT = os.path.expanduser("~/ComfyUI/output/video10s")
frames_all = np.load("/tmp/v5b_frames.npy")
assert frames_all.shape[0] == 241

BASE_FRAMES, SMOOTH = 60, 48


def grade(frames, target=None):
    hsvs = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32) for f in frames]
    s = np.array([h[:, :, 1].mean() / 255.0 for h in hsvs])
    tgt = target if target is not None else float(s[:BASE_FRAMES].mean())
    gain = np.clip(tgt / np.maximum(s, 1e-6), 0.75, 1.15)
    k = SMOOTH
    pad = np.pad(gain, (k // 2, k - 1 - k // 2), mode="edge")
    gain_s = np.convolve(pad, np.ones(k) / k, mode="valid")
    out = []
    for i, h in enumerate(hsvs):
        h2 = h.copy()
        h2[:, :, 1] = np.clip(h2[:, :, 1] * gain_s[i], 0, 255)
        out.append(cv2.cvtColor(h2.astype(np.uint8), cv2.COLOR_HSV2RGB))
    print(f"grade: target={tgt:.3f} gain {gain_s.min():.3f}-{gain_s.max():.3f} "
          f"sat {s.mean():.3f} -> {np.mean([h[:,:,1].mean()/255 for h in [cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32) for f in out]]):.3f}")
    return out


def write_mp4(frames, path, fps):
    import av
    from fractions import Fraction
    c = av.open(path, mode="w")
    st = c.add_stream("libx264", rate=Fraction(fps, 1))
    st.width, st.height = frames[0].shape[1], frames[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.options = {"crf": "16", "preset": "medium"}
    tb = Fraction(1, fps)
    for i, f in enumerate(frames):
        vf = av.VideoFrame.from_ndarray(f, format="rgb24")
        vf.pts = i
        vf.time_base = tb
        for pkt in st.encode(vf):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    print("mp4:", path, round(os.path.getsize(path) / 2**20, 2), "MB")


def write_webp(frames, path, fps):
    imgs = [Image.fromarray(f) for f in frames]
    dur = int(round(1000 / fps))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                 lossless=True, quality=100)
    print("webp:", path, round(os.path.getsize(path) / 2**20, 2), "MB")


# (a) 原始链式输出 241 帧 @24fps
write_mp4(list(frames_all), os.path.join(PROJ, "short10s_chain_241f_24fps.mp4"), 24)

# (b) 交付版：前 210 帧 @21fps=10.000s + 配平
core = list(frames_all[:210])
graded = grade(core)
write_mp4(graded, os.path.join(PROJ, "short10s_final_10s.mp4"), 21)
write_webp(graded, os.path.join(PROJ, "short10s_final_10s_lossless.webp"), 21)

# 抽样帧（每 30 帧一张 + 尾帧）
for i in [0, 60, 120, 180, 209]:
    Image.fromarray(graded[i]).save(os.path.join(CHECK, f"final10s_f{i:03d}.png"))

# 拷贝到 ComfyUI 输出目录，便于网页预览
import shutil
for f in ["short10s_final_10s.mp4", "short10s_chain_241f_24fps.mp4"]:
    shutil.copyfile(os.path.join(PROJ, f), os.path.join(OUT, f))
print("copied to ComfyUI output/video10s/")
print("done: 210 frames @21fps =", 210 / 21.0, "s")