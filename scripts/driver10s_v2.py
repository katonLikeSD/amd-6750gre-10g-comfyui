#!/usr/bin/env python3
"""10s 短片 v2：DS 修订提示词 + 三段链式（裁掉 I2V 段尾部退化区）

结构：seg1 121帧(T2V) + seg2 121帧取前90 + seg3 61帧取前30 = 241帧 = 10.04s@24fps
依据：v1 实测 seg2(I2V) 从约第 60 帧起过饱和/糊化 → 每段只用前 90/30 帧
"""
import json, os, shutil, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/video10s"))
import driver10s as D

POS_V2 = ("A cinematic slow lateral tracking shot moving right: an orange cat walking slowly "
          "through sunlit green grass in a park at golden hour. A gentle breeze sways the "
          "grass and leaves; the cat's tail swings gently as it walks. Warm soft sunlight, "
          "slightly blurred background, steady smooth camera movement, clean composition, "
          "high quality, detailed.")
NEG_V2 = "blurry, low quality, deformed, flickering, watermark, text"
D.POS, D.NEG = POS_V2, NEG_V2

FULL, STEPS = 121, 35
USE2, SEG3_F, USE3 = 90, 61, 30     # seg2 用前 90 帧；seg3 生成 61 帧用前 30 帧
PROJ, CHECK = D.PROJ, D.CHECK


def last_frame_png(webp, idx, dst_name):
    """从无损 webp 取第 idx 帧，写入 ComfyUI input/ 供 LoadImage 使用"""
    im = Image.open(webp)
    im.seek(idx)
    p = os.path.join(D.INP, dst_name)
    Image.fromarray(np.asarray(im.convert("RGB"))).save(p)
    D.log(f"抽帧 {idx} -> {p}")
    return dst_name


def main():
    D.preflight()
    st = {}

    # ---- seg1: T2V 121帧（用全部）----
    r1 = D.run_graph("v2_seg1", D.graph("v2_seg1", FULL, STEPS, last_frame=True), timeout=7200)
    w1 = D.out_path([f for f in r1["files"] if f["filename"].endswith(".webp")][0])
    lp = D.out_path([f for f in r1["files"] if f["filename"].endswith(".png")][0])
    a1 = D.load_webp(w1)
    st["seg1"] = {**r1, "webp": w1, "video": D.frame_stats(a1)}

    # ---- seg2: I2V from seg1[120] ----
    shutil.copyfile(lp, os.path.join(D.INP, "v2_seg1_last.png"))
    r2 = D.run_graph("v2_seg2", D.graph("v2_seg2", FULL, STEPS, start_image="v2_seg1_last.png"),
                     timeout=7200)
    w2 = D.out_path([f for f in r2["files"] if f["filename"].endswith(".webp")][0])
    a2 = D.load_webp(w2)
    st["seg2"] = {**r2, "webp": w2, "video": D.frame_stats(a2)}

    # ---- seg3: I2V from seg2[90]（用前 30 帧补足 241）----
    s3_name = last_frame_png(w2, USE2, "v2_seg2_f090.png")
    r3 = D.run_graph("v2_seg3", D.graph("v2_seg3", SEG3_F, STEPS, start_image=s3_name),
                     timeout=7200)
    w3 = D.out_path([f for f in r3["files"] if f["filename"].endswith(".webp")][0])
    a3 = D.load_webp(w3)
    st["seg3"] = {**r3, "webp": w3, "video": D.frame_stats(a3)}

    # ---- 拼接 241 帧 ----
    frames = np.concatenate([a1, a2[1:1 + USE2], a3[1:1 + USE3]])
    assert frames.shape[0] == 241, frames.shape

    d = lambda x, y: float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())
    m1 = float(np.median([d(a1[i], a1[i + 1]) for i in range(a1.shape[0] - 1)]))
    m2 = float(np.median([d(a2[i], a2[i + 1]) for i in range(USE2)]))
    m3 = float(np.median([d(a3[i], a3[i + 1]) for i in range(USE3)]))
    seamA = d(a1[-1], a2[0])
    seamB = d(a2[USE2], a3[0])

    def satof(arr):
        a = arr.astype(np.float32) / 255.0
        mx, mn = a.max(axis=2), a.min(axis=2)
        return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())

    report = {
        "seamA_absdiff": round(seamA, 2), "seamB_absdiff": round(seamB, 2),
        "median_interframe_seg1": round(m1, 2), "median_interframe_seg2(used)": round(m2, 2),
        "median_interframe_seg3(used)": round(m3, 2),
        "saturation_seg1": round(satof(a1), 3),
        "saturation_seg2_used": round(satof(a2[:USE2 + 1]), 3),
        "saturation_seg2_full_tail": round(satof(a2[-20:]), 3),
        "saturation_seg3_used": round(satof(a3[:USE3 + 1]), 3),
        "total_frames": 241, "duration_sec": 10.04,
    }

    import av
    from fractions import Fraction
    mp4 = os.path.join(PROJ, "short10s_v2_24fps.mp4")
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
    report["mp4"] = mp4
    report["mp4_mb"] = round(os.path.getsize(mp4) / 2**20, 2)

    webp_final = os.path.join(PROJ, "short10s_v2_lossless.webp")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(webp_final, save_all=True, append_images=imgs[1:],
                 duration=41, loop=0, lossless=True, quality=100)
    report["webp_mb"] = round(os.path.getsize(webp_final) / 2**20, 2)

    for i in [0, 60, 118, 119, 120, 121, 150, 200, 210, 211, 212, 240]:
        Image.fromarray(frames[i]).save(os.path.join(CHECK, f"v2_f{i:03d}.png"))
    with open(os.path.join(PROJ, "seam_report_v2.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(PROJ, "state_v2.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "video"} for k, v in st.items()},
                  f, ensure_ascii=False, indent=1)
    D.log("v2 完成:", json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()