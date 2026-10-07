#!/usr/bin/env python3
"""Generate test queue files for the limit tests."""
import json, sys

POS = ("A cute orange cat walking on green grass in a sunny park, "
       "smooth motion, high quality, detailed")
NEG = "blurry, low quality, distorted, flickering"

VAE = "wan2.2_vae.safetensors"
UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
GGUF_5B = "Wan2.2-TI2V-5B-Q4_K_M.gguf"
GGUF_SPMIX = "wan225bi2vspmixver021n_v21.gguf"

# latent token count for Wan2.2 5B: (W/16)*(H/16)*((F-1)//4+1)
def tokens(w, h, f):
    return (w // 16) * (h // 16) * ((f - 1) // 4 + 1)


def p1_graph(tid, w, h, length, tiled=True, tile=512, overlap=64, tsize=64, toverlap=8):
    g = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["1", 0], "width": w, "height": h,
                          "length": length, "batch_size": 1}},
    }
    if tiled:
        g["3"] = {"class_type": "VAEDecodeTiled",
                  "inputs": {"samples": ["2", 0], "vae": ["1", 0],
                              "tile_size": tile, "overlap": overlap,
                              "temporal_size": tsize, "temporal_overlap": toverlap}}
    else:
        g["3"] = {"class_type": "VAEDecode",
                  "inputs": {"samples": ["2", 0], "vae": ["1", 0]}}
    g["4"] = {"class_type": "SaveAnimatedWEBP",
              "inputs": {"images": ["3", 0], "filename_prefix": f"limitTest/{tid}",
                          "fps": 16.0, "lossless": True, "quality": 90,
                          "method": "default"}}
    return g


def p2_graph(tid, model, w, h, length, steps, cfg=5.0, tiled_params=None):
    g = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": UMT5, "type": "wan", "device": "cpu"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEG}},
        "5": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}},
        "6": {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["1", 0], "width": w, "height": h,
                          "length": length, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["5", 0], "positive": ["3", 0],
                          "negative": ["4", 0], "latent_image": ["6", 0],
                          "seed": 42, "steps": steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m_sde_heun",
                          "scheduler": "sgm_uniform", "denoise": 1.0}},
    }
    if tiled_params:
        g["8"] = {"class_type": "VAEDecodeTiled",
                  "inputs": {"samples": ["7", 0], "vae": ["1", 0], **tiled_params}}
    else:
        g["8"] = {"class_type": "VAEDecode",
                  "inputs": {"samples": ["7", 0], "vae": ["1", 0]}}
    g["9"] = {"class_type": "SaveAnimatedWEBP",
              "inputs": {"images": ["8", 0], "filename_prefix": f"limitTest/{tid}",
                          "fps": 16.0, "lossless": False, "quality": 90,
                          "method": "default"}}
    return g


def line(tid, graph, timeout):
    return json.dumps({"id": tid, "prompt": graph, "timeout_sec": timeout},
                      ensure_ascii=False)


# tile 参数阶梯：从默认到最保守（temporal_size 输入值为每块的像素帧数）
# 注：L0_untiled 已移除 —— untiled 解码在本机触发 ROCm 硬崩溃（ComfyUI 进程 Abort）
LADDER = [
    ("L1_512_64", dict(tiled=True, tile=512, overlap=64, tsize=64, toverlap=8)),
    ("L2_512_16", dict(tiled=True, tile=512, overlap=64, tsize=16, toverlap=8)),
    ("L3_512_8", dict(tiled=True, tile=512, overlap=64, tsize=8, toverlap=8)),
    ("L4_256_16", dict(tiled=True, tile=256, overlap=64, tsize=16, toverlap=8)),
    ("L5_256_8", dict(tiled=True, tile=256, overlap=64, tsize=8, toverlap=4)),
    ("L6_128_8", dict(tiled=True, tile=128, overlap=32, tsize=8, toverlap=4)),
]


def gen_phase1(path):
    res_list = [(832, 480), (480, 832), (704, 480), (1280, 720), (704, 1280)]
    frames = [17, 33, 49, 65, 81]
    combos = []
    for (w, h) in res_list:
        for f in frames:
            combos.append((w, h, f))
    combos.sort(key=lambda c: tokens(*c))  # ascending token count
    with open(path, "w") as fh:
        for (w, h, f) in combos:
            base = f"p1_{w}x{h}_f{f}"
            ladder = []
            for (tag, params) in LADDER:
                if tag == "L0_untiled" and f != 17:
                    continue
                g = p1_graph(base + tag, w, h, f, tiled=False) if params is None \
                    else p1_graph(base + tag, w, h, f, **params)
                ladder.append({"tag": tag, "prompt": g})
            fh.write(json.dumps({"id": base, "ladder": ladder, "timeout_sec": 900}) + "\n")


def gen_phase1_untiled(path, combos):
    with open(path, "w") as fh:
        for (w, h, f) in combos:
            tid = f"p1u_{w}x{h}_f{f}_untiled"
            fh.write(line(tid, p1_graph(tid, w, h, f, tiled=False), 900) + "\n")


def gen_phase2(path, tests):
    # tests: list of dicts {id, model, w, h, f, steps, cfg, tiled_params}
    with open(path, "w") as fh:
        for t in tests:
            g = p2_graph(t["id"], t["model"], t["w"], t["h"], t["f"],
                         t["steps"], t.get("cfg", 5.0), t.get("tiled_params"))
            fh.write(line(t["id"], g, t.get("timeout", 7200)) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    out = sys.argv[2]
    if mode == "phase1":
        gen_phase1(out)
    elif mode == "phase1u":
        combos = json.loads(sys.argv[3])  # [[w,h,f],...]
        gen_phase1_untiled(out, combos)
    elif mode == "phase2":
        gen_phase2(out, json.loads(sys.argv[3]))
    print("written", out)
