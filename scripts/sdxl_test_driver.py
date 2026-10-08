#!/usr/bin/env python3
import json, os, time, urllib.request, sys

API = "http://127.0.0.1:8188"

def post(path, data):
    req = urllib.request.Request(API+path, data=json.dumps(data).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(path):
    return json.loads(urllib.request.urlopen(API+path, timeout=30).read())

def wf(ckpt, pos, neg, w, h, seed, steps, cfg=7.0, sampler="dpmpp_2m_sde", sched="karras", prefix="sdxlTest/run"):
    return {
      "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt}},
      "2": {"class_type":"CLIPTextEncode","inputs":{"text":pos,"clip":["1",1]}},
      "3": {"class_type":"CLIPTextEncode","inputs":{"text":neg,"clip":["1",1]}},
      "4": {"class_type":"EmptyLatentImage","inputs":{"width":w,"height":h,"batch_size":1}},
      "5": {"class_type":"KSampler","inputs":{"seed":seed,"steps":steps,"cfg":cfg,
          "sampler_name":sampler,"scheduler":sched,"denoise":1.0,
          "model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0]}},
      "6": {"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
      "7": {"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":prefix}},
    }

POS = "masterpiece, best quality, 1girl, long hair, white dress, cherry blossom trees, spring, petals in wind, soft sunlight, detailed eyes, smiling"
NEG = "lowres, worst quality, low quality, bad anatomy, bad hands, missing fingers, watermark, text, blurry"

def run(name, w, h, seed, steps, cfg, sampler, sched):
    t_start = time.time()
    pid = post("/prompt", {"prompt": wf("novaAnimeXL_ilV190.safetensors", POS, NEG, w, h, seed, steps, cfg, sampler, sched, prefix=f"sdxlTest/{name}")})["prompt_id"]
    while True:
        time.sleep(2)
        h_ = get(f"/history/{pid}")
        if pid in h_:
            break
        if time.time()-t_start > 1500:
            print(json.dumps({"name":name,"status":"TIMEOUT"})); return
    entry = h_[pid]
    status = entry["status"]["status_str"]
    outputs = list(entry["outputs"].values())
    imgs = []
    for o in outputs:
        for im in o.get("images", []):
            imgs.append(im)
    rec = {"name":name,"wall_s":round(time.time()-t_start,1),"status":status,
           "seed":seed,"steps":steps,"cfg":cfg,"sampler":sampler,"scheduler":sched,
           "resolution":f"{w}x{h}","outputs":imgs}
    print(json.dumps(rec, ensure_ascii=False))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sdxl_results.jsonl"), "a") as f:
        f.write(json.dumps(rec)+"\n")

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        int(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8])
