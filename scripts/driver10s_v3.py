#!/usr/bin/env python3
"""10s 短片 v3：复用 v2 的 seg1(121帧) + seg2 前55帧，只新跑两个 61 帧短段

依据 v2 实测：I2V 续写段从约第 86 帧起出现渐进辉光/过曝；条件帧落在污染区会让下一段立刻退化。
v3 规则：每段只取前 30-55 帧（安全区），条件帧一律取干净区；提示词三段完全相同（DS 建议）。
结构：a1[0..120] + a2[1..55] + b3[1..30] + b4[1..35] = 121+55+30+35 = 241 帧 = 10.04s@24fps
"""
import json, os, shutil, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/video10s"))
import driver10s as D
import driver10s_v2 as V2

FULL, SHORT, STEPS = 121, 61, 35
USE2, USE3, USE4 = 55, 30, 35
PROJ, CHECK = D.PROJ, D.CHECK


def grab(webp, idx, name):
    im = Image.open(webp)
    im.seek(idx)
    p = os.path.join(D.INP, name)
    Image.fromarray(np.asarray(im.convert("RGB"))).save(p)
    D.log(f"抽帧 {idx} -> {p}")
    return name


def main():
    D.preflight()
    a1 = D.load_webp(os.path.join(D.OUT, "video10s/v2_seg1_00001_.webp"))
    a2 = D.load_webp(os.path.join(D.OUT, "video10s/v2_seg2_00001_.webp"))
    assert a1.shape[0] == FULL and a2.shape[0] == FULL

    s3 = grab(os.path.join(D.OUT, "video10s/v2_seg2_00001_.webp"), USE2, "v3_seg3_start.png")
    r3 = D.run_graph("v3_seg3", D.graph("v3_seg3", SHORT, STEPS, start_image=s3), timeout=3600)
    w3 = D.out_path([f for f in r3["files"] if f["filename"].endswith(".webp")][0])
    b3 = D.load_webp(w3)
    D.log("seg3' 帧统计:", json.dumps(D.frame_stats(b3), ensure_ascii=False))

    s4 = grab(w3, USE3, "v3_seg4_start.png")
    r4 = D.run_graph("v3_seg4", D.graph("v3_seg4", SHORT, STEPS, start_image=s4), timeout=3600)
    w4 = D.out_path([f for f in r4["files"] if f["filename"].endswith(".webp")][0])
    b4 = D.load_webp(w4)
    D.log("seg4' 帧统计:", json.dumps(D.frame_stats(b4), ensure_ascii=False))

    frames = np.concatenate([a1, a2[1:1 + USE2], b3[1:1 + USE3], b4[1:1 + USE4]])
    assert frames.shape[0] == 241, frames.shape
    allframes = np.concatenate([a1, a2, b3, b4])   # 供逐段质检

    def d(x, y):
        return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

    def bloom(arr):
        g = arr.astype(np.float32).mean(axis=3) / 255.0
        return float((g > 0.97).mean())

    def sat(arr):
        a = arr.astype(np.float32) / 255.0
        mx, mn = a.max(axis=3), a.min(axis=3)
        return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())

    report = {
        "seam_ab": round(d(a1[-1], a2[0]), 2),
        "seam_bc": round(d(a2[USE2], b3[0]), 2),
        "seam_cd": round(d(b3[USE3], b4[0]), 2),
        "median_interframe_seg1": round(float(np.median([d(a1[i], a1[i+1]) for i in range(FULL-1)])), 2),
        "bloom_pct_seg2_frame0_55": round(bloom(a2[:USE2+1]) * 100, 2),
        "bloom_pct_seg2_frame86_121": round(bloom(a2[86:]) * 100, 2),
        "bloom_pct_seg3_used": round(bloom(b3[:USE3+1]) * 100, 2),
        "bloom_pct_seg3_full": round(bloom(b3) * 100, 2),
        "bloom_pct_seg4_used": round(bloom(b4[:USE4+1]) * 100, 2),
        "bloom_pct_seg4_full": round(bloom(b4) * 100, 2),
        "sat_seg1": round(sat(a1), 3),
        "sat_seg3_used": round(sat(b3[:USE3+1]), 3),
        "sat_seg4_used": round(sat(b4[:USE4+1]), 3),
        "total_frames": 241, "duration_sec": 10.04,
    }

    import av
    from fractions import Fraction
    mp4 = os.path.join(PROJ, "short10s_v3_24fps.mp4")
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

    webp_final = os.path.join(PROJ, "short10s_v3_lossless.webp")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(webp_final, save_all=True, append_images=imgs[1:], duration=41,
                 loop=0, lossless=True, quality=100)
    report["webp_mb"] = round(os.path.getsize(webp_final) / 2**20, 2)

    for i in [0, 60, 120, 121, 150, 175, 176, 177, 205, 206, 210, 236, 240]:
        Image.fromarray(frames[i]).save(os.path.join(CHECK, f"v3_f{i:03d}.png"))

    # 逐段bloom曲线（每5帧）便于回报 DS
    curve = []
    for seg, arr, n in (("a1", a1, FULL), ("a2", a2, FULL), ("b3", b3, SHORT), ("b4", b4, SHORT)):
        for i in range(0, n, 5):
            curve.append({"seg": seg, "f": i, "bloom": round(bloom(arr[i:i+1]) * 100, 3)})
    report["bloom_curve_every5"] = curve

    with open(os.path.join(PROJ, "seam_report_v3.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    D.log("v3 完成:", json.dumps({k: v for k, v in report.items() if k != "bloom_curve_every5"},
                                 ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()