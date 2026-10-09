#!/usr/bin/env python3
"""10s 短片 v5：复用 v4 素材（seg1 全 + seg2/seg3 前 35 帧），新跑两段 61 帧（用 35/15）

依据 v4 实测：每段安全深度约 44-50 帧；条件帧若越过上一段安全区，退化会被继承放大。
v5 结构：a1[0..120] + a2[1..35] + a3[1..35] + b5[1..35] + b6[1..15] = 241 帧 = 10.04s@24fps
其中 b5 条件帧 = a3[35]，b6 条件帧 = b5[35]（均取段内 35 帧处，安全区内）。
"""
import json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/video10s"))
import driver10s as D
import driver10s_v4 as V4

D.POS, D.NEG = V4.POS_V4, V4.NEG_V4
D.STEPS_FULL = 35
OUT, PROJ, CHECK = D.OUT, D.PROJ, D.CHECK
SHORT, STEPS = 61, 35
U2 = U3 = U5 = 35
U6 = 15
W1 = os.path.join(OUT, "video10s/v4_seg1_00001_.webp")
W2 = os.path.join(OUT, "video10s/v4_seg2_00001_.webp")
W3 = os.path.join(OUT, "video10s/v4_seg3_00001_.webp")


def grab(webp, idx, name):
    im = Image.open(webp)
    im.seek(idx)
    p = os.path.join(D.INP, name)
    Image.fromarray(np.asarray(im.convert("RGB"))).save(p)
    D.log(f"抽帧 {idx} -> {p}")
    return name


def sat(arr):
    a = arr.astype(np.float32) / 255.0
    mx, mn = a.max(axis=3), a.min(axis=3)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())


def bloom(arr):
    g = arr.astype(np.float32).mean(axis=3) / 255.0
    return float((g > 0.97).mean())


def main():
    D.preflight()
    a1 = D.load_webp(W1)
    a2 = D.load_webp(W2)
    a3 = D.load_webp(W3)

    s5 = grab(W3, U3, "v5_seg5_start.png")
    r5 = D.run_graph("v5_seg5", D.graph("v5_seg5", SHORT, STEPS, start_image=s5), timeout=3600)
    w5 = D.out_path([f for f in r5["files"] if f["filename"].endswith(".webp")][0])
    b5 = D.load_webp(w5)
    D.log("seg5 曲线:", json.dumps([{"f": i, "sat": round(sat(b5[i:i+1]), 3),
                                     "bloom": round(bloom(b5[i:i+1]) * 100, 2)}
                                    for i in range(0, SHORT, 5)], ensure_ascii=False))

    s6 = grab(w5, U5, "v5_seg6_start.png")
    r6 = D.run_graph("v5_seg6", D.graph("v5_seg6", SHORT, STEPS, start_image=s6), timeout=3600)
    w6 = D.out_path([f for f in r6["files"] if f["filename"].endswith(".webp")][0])
    b6 = D.load_webp(w6)
    D.log("seg6 曲线:", json.dumps([{"f": i, "sat": round(sat(b6[i:i+1]), 3),
                                     "bloom": round(bloom(b6[i:i+1]) * 100, 2)}
                                    for i in range(0, SHORT, 5)], ensure_ascii=False))

    frames = np.concatenate([a1, a2[1:1+U2], a3[1:1+U3], b5[1:1+U5], b6[1:1+U6]])
    assert frames.shape[0] == 241, frames.shape

    def d(x, y):
        return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

    rep = {
        "seams": [round(d(a1[-1], a2[0]), 2), round(d(a2[U2], a3[0]), 2),
                  round(d(a3[U3], b5[0]), 2), round(d(b5[U5], b6[0]), 2)],
        "sat_per_seg_used": [round(sat(a1), 3), round(sat(a2[:U2+1]), 3), round(sat(a3[:U3+1]), 3),
                             round(sat(b5[:U5+1]), 3), round(sat(b6[:U6+1]), 3)],
        "bloom_per_seg_used": [round(bloom(a1)*100, 2), round(bloom(a2[:U2+1])*100, 2),
                               round(bloom(a3[:U3+1])*100, 2), round(bloom(b5[:U5+1])*100, 2),
                               round(bloom(b6[:U6+1])*100, 2)],
        "total_frames": 241, "duration_sec": 10.04,
    }
    import av
    from fractions import Fraction
    mp4 = os.path.join(PROJ, "short10s_v5_24fps.mp4")
    c = av.open(mp4, mode="w")
    st = c.add_stream("libx264", rate=Fraction(24, 1))
    st.width, st.height = frames.shape[2], frames.shape[1]
    st.pix_fmt = "yuv420p"
    st.options = {"crf": "16", "preset": "medium"}
    for i in range(frames.shape[0]):
        vf = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        vf.pts = i
        vf.time_base = Fraction(1, 24)
        for pkt in st.encode(vf):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    rep["mp4_mb"] = round(os.path.getsize(mp4) / 2**20, 2)
    webp_final = os.path.join(PROJ, "short10s_v5_lossless.webp")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(webp_final, save_all=True, append_images=imgs[1:], duration=41,
                 loop=0, lossless=True, quality=100)
    for i in [0, 60, 120, 121, 155, 156, 190, 191, 225, 226, 236, 240]:
        Image.fromarray(frames[i]).save(os.path.join(CHECK, f"v5_f{i:03d}.png"))
    with open(os.path.join(PROJ, "seam_report_v5.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    D.log("v5 完成:", json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()