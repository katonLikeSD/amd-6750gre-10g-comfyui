#!/usr/bin/env python3
"""J: batch_size=2 @1024², 20 steps, with in-process 0.1s VRAM sampling."""
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

wfj = {
  "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"novaAnimeXL_ilV190.safetensors"}},
  "2": {"class_type":"CLIPTextEncode","inputs":{"text":POS,"clip":["1",1]}},
  "3": {"class_type":"CLIPTextEncode","inputs":{"text":NEG,"clip":["1",1]}},
  "4": {"class_type":"EmptyLatentImage","inputs":{"width":1024,"height":1024,"batch_size":2}},
  "5": {"class_type":"KSampler","inputs":{"seed":20261225,"steps":20,"cfg":7.0,
      "sampler_name":"dpmpp_2m","scheduler":"karras","denoise":1.0,
      "model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0]}},
  "6": {"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
  "7": {"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"sdxlTest/J_batch2"}},
}

th = threading.Thread(target=sample); th.start()
t0 = time.time()
pid = post("/prompt", {"prompt": wfj})["prompt_id"]
while time.time()-t0 < 600:
    time.sleep(2)
    if pid in get(f"/history/{pid}"):
        break
time.sleep(2)
stop.set(); th.join()
peak = max(samples) if samples else 0
print(json.dumps({"name":"J_batch2","wall_s":round(time.time()-t0,1),
                  "sysfs_peak_GB":round(peak/1024,2),"samples":len(samples)}))
