#!/usr/bin/env python3
"""10s 短片 v4：换抗辉光的镜头设计（柔光树荫静态近景 + 极慢推近）

v1-v3 实测：退化主因是"长摇镜走进逆光"——影片位置 ~7.5-8s 后大面积高光被烘成辉光并传染下一段。
v4 设计：无逆光、无长摇镜、小幅度主体动作（舔爪/尾巴轻摆/草叶微风），负向显式压制 flare/bloom/过曝。
结构：seg1 T2V 121帧(全用) + seg2 61帧(用50) + seg3 61帧(用50) + seg4 61帧(用20) = 241帧 = 10.04s@24fps
用法：python3 driver10s_v4.py seg1   # 先跑第一段并质检
      python3 driver10s_v4.py rest   # 续跑并把成片拼出来
"""
import json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/video10s"))
import driver10s as D

POS_V4 = ("A calm cinematic medium close-up: an orange cat sitting on green grass in the soft "
          "shade of a tree, calmly licking its front paw. A gentle breeze moves a few blades of "
          "grass; the cat's tail tip flicks gently. Soft diffused daylight, even exposure, "
          "clean uncluttered background, slightly blurred background, steady camera with a very "
          "slow push-in, high quality, detailed.")
NEG_V4 = ("blurry, low quality, deformed, flickering, watermark, text, "
          "lens flare, bloom, overexposed, blown highlights, glow, hazy")
D.POS, D.NEG = POS_V4, NEG_V4
D.STEPS_FULL = 35

FULL, SHORT, STEPS = 121, 61, 35
U2, U3, U4 = 50, 50, 20
PROJ, CHECK, OUT = D.PROJ, D.CHECK, D.OUT
W1 = os.path.join(OUT, "video10s/v4_seg1_00001_.webp")


def grab(webp, idx, name):
    im = Image.open(webp)
    im.seek(idx)
    p = os.path.join(D.INP, name)
    Image.fromarray(np.asarray(im.convert("RGB"))).save(p)
    D.log(f"抽帧 {idx} -> {p}")
    return name


def bloom(arr):
    g = arr.astype(np.float32).mean(axis=3) / 255.0
    return float((g > 0.97).mean())


def sat(arr):
    a = arr.astype(np.float32) / 255.0
    mx, mn = a.max(axis=3), a.min(axis=3)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def seg1():
    D.preflight()
    r = D.run_graph("v4_seg1", D.graph("v4_seg1", FULL, STEPS, last_frame=True), timeout=7200)
    w = D.out_path([f for f in r["files"] if f["filename"].endswith(".webp")][0])
    a = D.load_webp(w)
    D.log("seg1:", json.dumps(D.frame_stats(a), ensure_ascii=False))
    D.save_samples(a, [0, 40, 80, 110, 120], "v4_seg1")
    curve = [{"f": i, "bloom": round(bloom(a[i:i+1]) * 100, 3), "sat": round(sat(a[i:i+1]), 3)}
             for i in range(0, FULL, 10)]
    D.log("seg1 逐10帧 bloom/sat:", json.dumps(curve, ensure_ascii=False))
    with open(os.path.join(PROJ, "v4_seg1_curve.json"), "w") as f:
        json.dump(curve, f, ensure_ascii=False, indent=1)


def rest():
    D.preflight()
    a1 = D.load_webp(W1)
    assert a1.shape[0] == FULL

    s2 = grab(W1, FULL - 1, "v4_seg2_start.png")
    r2 = D.run_graph("v4_seg2", D.graph("v4_seg2", SHORT, STEPS, start_image=s2), timeout=3600)
    w2 = D.out_path([f for f in r2["files"] if f["filename"].endswith(".webp")][0])
    b2 = D.load_webp(w2)

    s3 = grab(w2, U2, "v4_seg3_start.png")
    r3 = D.run_graph("v4_seg3", D.graph("v4_seg3", SHORT, STEPS, start_image=s3), timeout=3600)
    w3 = D.out_path([f for f in r3["files"] if f["filename"].endswith(".webp")][0])
    b3 = D.load_webp(w3)

    s4 = grab(w3, U3, "v4_seg4_start.png")
    r4 = D.run_graph("v4_seg4", D.graph("v4_seg4", SHORT, STEPS, start_image=s4), timeout=3600)
    w4 = D.out_path([f for f in r4["files"] if f["filename"].endswith(".webp")][0])
    b4 = D.load_webp(w4)

    frames = np.concatenate([a1, b2[1:1 + U2], b3[1:1 + U3], b4[1:1 + U4]])
    assert frames.shape[0] == 241, frames.shape

    def d(x, y):
        return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

    rep = {
        "seam12": round(d(a1[-1], b2[0]), 2), "seam23": round(d(b2[U2], b3[0]), 2),
        "seam34": round(d(b3[U3], b4[0]), 2),
        "bloom_used": [round(bloom(b2[:U2+1])*100, 2), round(bloom(b3[:U3+1])*100, 2),
                       round(bloom(b4[:U4+1])*100, 2)],
        "sat_used": [round(sat(b2[:U2+1]), 3), round(sat(b3[:U3+1]), 3), round(sat(b4[:U4+1]), 3)],
        "total_frames": 241, "duration_sec": 10.04,
    }
    import av
    from fractions import Fraction
    mp4 = os.path.join(PROJ, "short10s_v4_24fps.mp4")
    c = av.open(mp4, mode="w")
    s = c.add_stream("libx264", rate=Fraction(24, 1))
    s.width, s.height = frames.shape[2], frames.shape[1]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "16", "preset": "medium"}
    for i in range(frames.shape[0]):
        vf = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        vf.pts = i
        vf.time_base = Fraction(1, 24)
        for pkt in s.encode(vf):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    rep["mp4_mb"] = round(os.path.getsize(mp4) / 2**20, 2)

    webp_final = os.path.join(PROJ, "short10s_v4_lossless.webp")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(webp_final, save_all=True, append_images=imgs[1:], duration=41,
                 loop=0, lossless=True, quality=100)
    for i in [0, 60, 120, 121, 170, 171, 172, 220, 221, 236, 240]:
        Image.fromarray(frames[i]).save(os.path.join(CHECK, f"v4_f{i:03d}.png"))
    curve = []
    for seg, arr, n in (("b2", b2, SHORT), ("b3", b3, SHORT), ("b4", b4, SHORT)):
        for i in range(0, n, 5):
            curve.append({"seg": seg, "f": i, "bloom": round(bloom(arr[i:i+1])*100, 3)})
    rep["curve"] = curve
    with open(os.path.join(PROJ, "seam_report_v4.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    D.log("v4 完成:", json.dumps({k: v for k, v in rep.items() if k != "curve"},
                                 ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"seg1": seg1, "rest": rest}[sys.argv[1]]()