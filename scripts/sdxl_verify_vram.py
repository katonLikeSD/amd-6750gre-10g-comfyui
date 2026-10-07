#!/usr/bin/env python3
"""One-shot VRAM verification: sampler thread + generation + summary, no background tasks."""
import glob, json, threading, time, urllib.request

API = "http://127.0.0.1:8188"
samples = []
stop = threading.Event()
paths = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))
vp = paths[0]

def sample():
    t0 = time.time()
    while not stop.is_set():
        try:
            v = int(open(vp).read().strip())
        except Exception:
            v = -1
        samples.append((round(time.time()-t0, 2), v/2**20))
        time.sleep(0.1)

def post(path, data):
    req = urllib.request.Request(API+path, data=json.dumps(data).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(path):
    return json.loads(urllib.request.urlopen(API+path, timeout=30).read())

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdxl_test_driver import wf

POS = "masterpiece, best quality, 1girl, cat ears, cafe window, rainy day, warm light, detailed eyes"
NEG = "lowres, worst quality, bad anatomy, watermark"

th = threading.Thread(target=sample); th.start()
pid = post("/prompt", {"prompt": wf("novaAnimeXL_ilV190.safetensors", POS, NEG, 1024, 1024, 20261011, 15, 7.0, "dpmpp_2m_sde", "karras", prefix="sdxlTest/run4_verify")})["prompt_id"]
t0 = time.time()
phases = {}
while time.time() - t0 < 900:
    time.sleep(1)
    h = get(f"/history/{pid}")
    if pid in h:
        break
# peek torch's view right after done, then a couple of idle samples to confirm release
time.sleep(2)
stop.set(); th.join()
peak = max(v for _, v in samples if v >= 0)
print(json.dumps({
    "prompt_id": pid,
    "wall_s": round(time.time()-t0, 1),
    "sysfs_peak_MB": round(peak, 0),
    "samples": len(samples),
    "timeline_every_1s": [(t, v) for t, v in samples if abs((t % 1)) < 0.06 and v >= 0][:400],
    "history_status": list(h[pid]["status"].get("status_str") for _ in [0]),
}))
# exact per-node execution timing from history meta
m = h[pid].get("meta", {})
ks = m.get("5", {})
print("KSampler meta:", json.dumps(ks))
