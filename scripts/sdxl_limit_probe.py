#!/usr/bin/env python3
"""I2: 1216x1824 with VAEDecodeTiled; K: 1536x1536 untiled. In-process 0.1s VRAM sampling."""
import glob, json, threading, time, urllib.request

API = "http://127.0.0.1:8188"
samples = []
stop = threading.Event()
vp = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))[0]

def sample():
    while not stop.is_set():
        try:
            samples.append(int(open(vp).read().strip())/2**20)
        except Exception:
            pass
        time.sleep(0.1)

def post(path, data):
    req = urllib.request.Request(API+path, data=json.dumps(data).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(path):
    return json.loads(urllib.request.urlopen(API+path, timeout=30).read())

POS = "masterpiece, best quality, 1girl, long hair, white dress, cherry blossom trees, spring, petals in wind, soft sunlight, detailed eyes, smiling"
NEG = "lowres, worst quality, low quality, bad anatomy, bad hands, missing fingers, watermark, text, blurry"

def wf(w, h, seed, steps, tiled):
    dec = ({"class_type":"VAEDecodeTiled","inputs":{"samples":["5",0],"vae":["1",2],"tile_size":512,"overlap":64,"temporal_size":64,"temporal_overlap":8}}
           if tiled else
           {"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}})
    d = {
      "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"novaAnimeXL_ilV190.safetensors"}},
      "2": {"class_type":"CLIPTextEncode","inputs":{"text":POS,"clip":["1",1]}},
      "3": {"class_type":"CLIPTextEncode","inputs":{"text":NEG,"clip":["1",1]}},
      "4": {"class_type":"EmptyLatentImage","inputs":{"width":w,"height":h,"batch_size":1}},
      "5": {"class_type":"KSampler","inputs":{"seed":seed,"steps":steps,"cfg":7.0,
          "sampler_name":"dpmpp_2m","scheduler":"karras","denoise":1.0,
          "model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0]}},
      "6": dec,
      "7": {"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":f"sdxlTest/{'I2' if tiled else 'K'}_untiled"}},
    }
    if tiled:
        d["7"]["inputs"]["filename_prefix"] = "sdxlTest/I2_tiled"
    return d

def run(name, w, h, seed, steps, tiled):
    samples.clear()
    t0 = time.time()
    try:
        pid = post("/prompt", {"prompt": wf(w, h, seed, steps, tiled)})["prompt_id"]
    except Exception as e:
        print(json.dumps({"name":name,"status":"POST_FAIL","err":str(e)})); return
    ok = False
    while time.time()-t0 < 600:
        time.sleep(2)
        try:
            h_ = get(f"/history/{pid}")
        except Exception:
            continue
        if pid in h_:
            ok = h_[pid]["status"]["status_str"] == "success"
            break
    peak = max(samples) if samples else 0
    print(json.dumps({"name":name,"wall_s":round(time.time()-t0,1),"ok":ok,
                      "sysfs_peak_GB":round(peak/1024,2),"res":f"{w}x{h}","tiled":tiled}))

th = threading.Thread(target=sample, daemon=True); th.start()
run("I2_tiled_1216x1824", 1216, 1824, 20261226, 25, True)
run("K_untiled_1536x1536", 1536, 1536, 20261227, 25, False)
stop.set()
# brightness check of the two new outputs
from PIL import Image
import numpy as np
for f in sorted(glob.glob("/home/jjt/ComfyUI/output/sdxlTest/I2_tiled*.png") + glob.glob("/home/jjt/ComfyUI/output/sdxlTest/K_untiled*.png")):
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
    print(f"{f.split('/')[-1]}: mean={a.mean():.1f} std={a.std():.1f}")
