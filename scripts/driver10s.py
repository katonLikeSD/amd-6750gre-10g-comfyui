#!/usr/bin/env python3
"""10s 高质量短片驱动（6750GRE 10G / Wan2.2 TI2V-5B GGUF / ComfyUI API）

阶段: draft -> seg1 -> seg2 -> assemble
用法: python3 driver10s.py <stage>
状态写 ~/video10s/state.json；抽样帧写 ~/video10s/check/；接缝报告写 ~/video10s/seam_report.json
"""
import json, os, shutil, sys, threading, time, urllib.request
import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8188"
COMFY = os.path.expanduser("~/ComfyUI")
OUT = os.path.join(COMFY, "output")
INP = os.path.join(COMFY, "input")
PROJ = os.path.expanduser("~/video10s")
CHECK = os.path.join(PROJ, "check")
STATE = os.path.join(PROJ, "state.json")
os.makedirs(CHECK, exist_ok=True)

VAE = "wan2.2_vae.safetensors"
UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
TI2V = "Wan2.2-TI2V-5B-Q4_K_M.gguf"
SPMIX = "wan225bi2vspmixver021n_v21.gguf"

POS = ("A cinematic slow lateral tracking shot: a cute orange cat walking slowly through "
       "sunlit green grass in a park at golden hour. A gentle breeze sways the grass and "
       "leaves; the cat's tail sways as it walks. Warm soft sunlight, shallow depth of "
       "field, steady smooth camera movement, clean composition, high quality, detailed.")
SAME = (" Keep exactly the same composition, lighting and camera motion as the previous "
        "frame; the motion continues without accelerating.")
NEG = "blurry, low quality, distorted, flickering, deformed"
TILED = {"tile_size": 128, "overlap": 32, "temporal_size": 8, "temporal_overlap": 4}

W, H = 704, 480
FPS = 24.0
FULL_FRAMES = 121      # 4n+1
DRAFT_FRAMES = 17
STEPS_FULL = 35
STEPS_DRAFT = 18
SEED = 42

REQUIRED_NODES = ["VAELoader", "CLIPLoader", "CLIPTextEncode", "UnetLoaderGGUF",
                  "Wan22ImageToVideoLatent", "KSampler", "VAEDecodeTiled",
                  "SaveAnimatedWEBP", "ImageFromBatch", "SaveImage", "LoadImage"]

VRAM_USED = "/sys/class/drm/card1/device/mem_info_vram_used"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def http_get(path, soft=False):
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        if soft:
            return {}
        raise


def http_post(path, data):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return json.loads(body) if body else {}


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(s):
    with open(STATE, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)


class VramMon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = None
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                with open(VRAM_USED) as f:
                    used = int(f.read().strip())
                if self.peak is None or used > self.peak:
                    self.peak = used
            except Exception:
                pass
            time.sleep(0.1)


def graph(tag, length, steps, seed=SEED, model=TI2V, start_image=None,
          pos=POS, fps=FPS, last_frame=False, mid_png=False):
    g = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": UMT5, "type": "wan", "device": "cpu"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEG}},
        "5": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}},
    }
    lv_in = {"vae": ["1", 0], "width": W, "height": H, "length": length, "batch_size": 1}
    if start_image:
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
        lv_in["start_image"] = ["10", 0]
    g["6"] = {"class_type": "Wan22ImageToVideoLatent", "inputs": lv_in}
    g["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["5", 0], "positive": ["3", 0], "negative": ["4", 0],
        "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 5.0,
        "sampler_name": "dpmpp_2m_sde_heun", "scheduler": "sgm_uniform", "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["1", 0], **TILED}}
    g["9"] = {"class_type": "SaveAnimatedWEBP", "inputs": {
        "images": ["8", 0], "filename_prefix": "video10s/" + tag,
        "fps": fps, "lossless": True, "quality": 100, "method": "default"}}
    if last_frame:
        g["11"] = {"class_type": "ImageFromBatch", "inputs": {
            "image": ["8", 0], "batch_index": length - 1, "length": 1}}
        g["12"] = {"class_type": "SaveImage", "inputs": {
            "images": ["11", 0], "filename_prefix": "video10s/" + tag + "_last"}}
    if mid_png:
        g["13"] = {"class_type": "ImageFromBatch", "inputs": {
            "image": ["8", 0], "batch_index": length // 2, "length": 1}}
        g["14"] = {"class_type": "SaveImage", "inputs": {
            "images": ["13", 0], "filename_prefix": "video10s/" + tag + "_mid"}}
    return g


def preflight():
    missing = [n for n in REQUIRED_NODES if n not in http_get("/object_info/" + n, soft=True)]
    if missing:
        raise SystemExit("缺少节点: " + ", ".join(missing))
    log("preflight ok: 11 nodes, 服务器就绪")


def run_graph(tag, g, timeout):
    mon = VramMon(); mon.start()
    t0 = time.time()
    r = http_post("/prompt", {"prompt": g, "client_id": "video10s"})
    pid = r.get("prompt_id")
    if not pid:
        mon.stop_flag = True
        raise SystemExit("提交失败: " + json.dumps(r)[:500])
    log(f"{tag} 已提交 prompt_id={pid}")
    last_print = 0
    hist = None
    while time.time() - t0 < timeout:
        time.sleep(2)
        h = http_get(f"/history/{pid}", soft=True)
        if pid in h:
            hist = h[pid]
            break
        if time.time() - last_print > 60:
            last_print = time.time()
            q = http_get("/queue", soft=True)
            running = len(q.get("queue_running", [])) if q else "?"
            peak = mon.peak / 2**30 if mon.peak else 0
            log(f"{tag} 进行中 {int(time.time()-t0)}s 队列运行中={running} 显存峰值={peak:.2f}G")
    mon.stop_flag = True
    wall = time.time() - t0
    if hist is None:
        raise SystemExit(f"{tag} 超时({timeout}s)")
    status = hist.get("status", {})
    files = []
    for node_out in hist.get("outputs", {}).values():
        for key in ("images", "gifs", "videos"):
            for item in node_out.get(key, []):
                files.append(item)
    peak = mon.peak / 2**30 if mon.peak else None
    log(f"{tag} 完成 {wall:.0f}s 峰值={peak:.2f}G status={status.get('status_str')}")
    if status.get("status_str") != "success":
        log("!! 执行状态异常:", json.dumps(status)[:800])
    log("输出文件:", json.dumps(files, ensure_ascii=False))
    return {"tag": tag, "wall_sec": round(wall, 1), "peak_vram_gb": round(peak, 2) if peak else None,
            "status": status.get("status_str"), "files": files}


def out_path(item):
    return os.path.join(OUT, item.get("subfolder", ""), item["filename"])


def load_webp(path):
    im = Image.open(path)
    n = getattr(im, "n_frames", 1)
    arrs = []
    for i in range(n):
        im.seek(i)
        arrs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return np.stack(arrs)


def frame_stats(arr):
    m = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return {"n": int(arr.shape[0]), "brightness_min": round(float(m.min()), 2),
            "brightness_max": round(float(m.max()), 2),
            "black_frames": int((m < 3).sum()), "nan": bool(np.isnan(arr).any())}


def save_samples(arr, indices, prefix):
    paths = []
    for i in indices:
        p = os.path.join(CHECK, f"{prefix}_f{i:03d}.png")
        Image.fromarray(arr[i]).save(p)
        paths.append(p)
    return paths


def cmd_draft():
    preflight()
    st = load_state()
    r = run_graph("draft17", graph("draft17", DRAFT_FRAMES, STEPS_DRAFT), timeout=1800)
    webp = [f for f in r["files"] if f["filename"].endswith(".webp")][0]
    arr = load_webp(out_path(webp))
    r["video"] = frame_stats(arr)
    r["sample_pngs"] = save_samples(arr, [0, DRAFT_FRAMES // 2, DRAFT_FRAMES - 1], "draft17")
    st["draft"] = r
    save_state(st)
    log("draft:", json.dumps(r["video"], ensure_ascii=False))
    log("抽样帧:", r["sample_pngs"])


def cmd_seg1():
    preflight()
    st = load_state()
    r = run_graph("seg1", graph("seg1", FULL_FRAMES, STEPS_FULL, last_frame=True), timeout=7200)
    webp = [f for f in r["files"] if f["filename"].endswith(".webp")][0]
    last = [f for f in r["files"] if f["filename"].endswith(".png")][0]
    arr = load_webp(out_path(webp))
    r["video"] = frame_stats(arr)
    r["webp"] = webp
    r["last_png"] = last
    r["sample_pngs"] = save_samples(arr, [0, 60, 120], "seg1")
    st["seg1"] = r
    save_state(st)
    log("seg1:", json.dumps(r["video"], ensure_ascii=False))
    log("尾帧:", out_path(last))


def cmd_seg2():
    preflight()
    st = load_state()
    src = out_path(st["seg1"]["last_png"])
    dst = os.path.join(INP, "seg1_last.png")
    shutil.copyfile(src, dst)
    log("尾帧已拷入 input:", dst)
    g = graph("seg2", FULL_FRAMES, STEPS_FULL, start_image="seg1_last.png",
              pos=POS + SAME, last_frame=False)
    r = run_graph("seg2", g, timeout=7200)
    webp = [f for f in r["files"] if f["filename"].endswith(".webp")][0]
    arr = load_webp(out_path(webp))
    r["video"] = frame_stats(arr)
    r["webp"] = webp
    r["sample_pngs"] = save_samples(arr, [0, 1, 60, 120], "seg2")
    st["seg2"] = r
    save_state(st)
    log("seg2:", json.dumps(r["video"], ensure_ascii=False))


def cmd_assemble():
    st = load_state()
    a1 = load_webp(out_path(st["seg1"]["webp"]))
    a2 = load_webp(out_path(st["seg2"]["webp"]))
    assert a1.shape[0] == FULL_FRAMES and a2.shape[0] == FULL_FRAMES, (a1.shape, a2.shape)

    d = lambda x, y: float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())
    seam_pix = d(a1[-1], a2[0])                                   # 尾帧 vs seg2首帧 像素差
    int1 = float(np.median([d(a1[i], a1[i + 1]) for i in range(a1.shape[0] - 1)]))
    int2 = float(np.median([d(a2[i], a2[i + 1]) for i in range(a2.shape[0] - 1)]))
    seam_next = d(a2[0], a2[1])
    b1 = a1.reshape(a1.shape[0], -1).mean(axis=1)
    b2 = a2.reshape(a2.shape[0], -1).mean(axis=1)
    seam_bri = float(abs(b1[-1] - b2[0]))

    report = {"seam_absdiff_last_vs_first": round(seam_pix, 2),
              "median_interframe_seg1": round(int1, 2),
              "median_interframe_seg2": round(int2, 2),
              "seg2_first_to_second": round(seam_next, 2),
              "seam_brightness_jump": round(seam_bri, 2)}
    # 241 帧：seg1 全 + seg2[1:]（丢弃共享首帧）；若接缝像素差显著大于段内帧差则记录
    use240 = seam_pix > max(2.0, 1.8 * max(int1, int2))
    report["decision"] = "240 (丢双过渡帧)" if use240 else "241 (seg1+seg2[1:])"
    frames = np.concatenate([a1, a2[1:]]) if not use240 else np.concatenate([a1[:-1], a2[1:]])
    report["total_frames"] = int(frames.shape[0])
    report["duration_sec"] = round(frames.shape[0] / FPS, 2)
    report["brightness_min"] = round(float(frames.reshape(frames.shape[0], -1).mean(axis=1).min()), 2)
    report["brightness_max"] = round(float(frames.reshape(frames.shape[0], -1).mean(axis=1).max()), 2)

    import av
    from fractions import Fraction
    mp4 = os.path.join(PROJ, "short10s_24fps.mp4")
    rate = Fraction(int(round(FPS)), 1)
    container = av.open(mp4, mode="w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height = frames.shape[2], frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "16", "preset": "medium"}
    for i in range(frames.shape[0]):
        vf = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        vf.pts = i
        vf.time_base = Fraction(1, int(round(FPS)))
        for pkt in stream.encode(vf):
            container.mux(pkt)
        if i % 60 == 0:
            log(f"mp4 编码 {i}/{frames.shape[0]}")
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    report["mp4"] = mp4
    report["mp4_mb"] = round(os.path.getsize(mp4) / 2**20, 2)

    webp_final = os.path.join(PROJ, "short10s_lossless.webp")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(webp_final, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / FPS), loop=0, lossless=True, quality=100)
    report["webp"] = webp_final
    report["webp_mb"] = round(os.path.getsize(webp_final) / 2**20, 2)

    # 接缝前后抽样帧（用于肉眼复核）
    seam_pngs = []
    for i in [a1.shape[0] - 3, a1.shape[0] - 1]:
        p = os.path.join(CHECK, f"seam_prev_f{i:03d}.png")
        Image.fromarray(a1[i]).save(p); seam_pngs.append(p)
    for i in [0, 1, 2, 3]:
        p = os.path.join(CHECK, f"seam_after_f{i:03d}.png")
        Image.fromarray(a2[i]).save(p); seam_pngs.append(p)
    report["seam_pngs"] = seam_pngs

    with open(os.path.join(PROJ, "seam_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log("assembled:", json.dumps(report, ensure_ascii=False, indent=1))


PROVEN_POS = ("A cute orange cat walking on green grass in a sunny park, "
              "smooth motion, high quality, detailed")


def cmd_probe():
    """探针: python3 driver10s.py probe <steps> [length] [mine|proven]"""
    preflight()
    steps = int(sys.argv[2])
    length = int(sys.argv[3]) if len(sys.argv) > 3 else DRAFT_FRAMES
    pmode = sys.argv[4] if len(sys.argv) > 4 else "mine"
    pos = POS if pmode == "mine" else PROVEN_POS
    tag = f"probe_s{steps}_f{length}_{pmode}"
    r = run_graph(tag, graph(tag, length, steps, pos=pos, mid_png=True), timeout=3600)
    webp = [f for f in r["files"] if f["filename"].endswith(".webp")][0]
    arr = load_webp(out_path(webp))
    r["video"] = frame_stats(arr)
    r["sample_pngs"] = save_samples(arr, [0, length // 2, length - 1], tag)
    r["mid_png_comfy"] = out_path([f for f in r["files"] if "mid" in f["filename"]][0])
    log(tag + ":", json.dumps(r["video"], ensure_ascii=False))
    log("抽样帧:", r["sample_pngs"])
    log("ComfyUI 原生 PNG:", r["mid_png_comfy"])


def cmd_pipeline():
    cmd_seg1()
    cmd_seg2()
    cmd_assemble()


if __name__ == "__main__":
    stage = sys.argv[1]
    {"draft": cmd_draft, "seg1": cmd_seg1, "seg2": cmd_seg2,
     "assemble": cmd_assemble, "pipeline": cmd_pipeline, "probe": cmd_probe}[stage]()