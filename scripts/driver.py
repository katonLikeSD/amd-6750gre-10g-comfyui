#!/usr/bin/env python3
"""ComfyUI limit-test driver: runs a queue of test graphs, records result rows.

Usage: python3 driver.py <queue.jsonl>
Each queue line: {"id": str, "prompt": <api-format graph>, "timeout_sec": int}
Appends one JSON result line per test to /tmp/comfy_test/results.jsonl
"""
import json, sys, time, threading, urllib.request, os
import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8188"
OUT_DIR = "/home/USER/ComfyUI/output"
RESULTS = "/tmp/comfy_test/results.jsonl"


def http_get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def http_post(path, data):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return json.loads(body) if body else {}


def disk_free_gb():
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 2**30


class VramMonitor(threading.Thread):
    """0.1s 间隔直读内核 sysfs 显存占用,避免 HTTP 轮询漏掉 100-300ms 的短峰."""

    VRAM_USED = "/sys/class/drm/card1/device/mem_info_vram_used"

    def __init__(self):
        super().__init__(daemon=True)
        self.max_used = None
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                with open(self.VRAM_USED) as f:
                    used = int(f.read().strip())
                if self.max_used is None or used > self.max_used:
                    self.max_used = used
            except Exception:
                pass
            time.sleep(0.1)


def analyze_webp(path):
    im = Image.open(path)
    n = getattr(im, "n_frames", 1)
    means, stds = [], []
    for i in range(n):
        im.seek(i)
        fr = np.asarray(im.convert("RGB"), dtype=np.float32)
        means.append(float(fr.mean()))
        stds.append(float(fr.std()))
    m = np.array(means, dtype=np.float64)
    return {
        "frames": n,
        "mean_brightness": round(float(np.nanmean(m)), 2),
        "black": bool(np.nanmin(m) < 3.0) if not np.isnan(m).all() else False,
        "nan": bool(np.isnan(m).any()),
        "frame_means_minmax": [round(float(np.nanmin(m)), 1), round(float(np.nanmax(m)), 1)],
        "mean_std": round(float(np.mean(stds)), 2),
    }


def run_one(test):
    attempts = test.get("ladder") or [{"tag": "", "prompt": test["prompt"]}]
    last = None
    for a in attempts:
        last = run_graph(test["id"] + a["tag"], a["prompt"],
                         test.get("timeout_sec", 900))
        last["id"] = test["id"]
        last["attempt_tag"] = a["tag"]
        if last["status"] in ("ok", "skip_no_disk", "submit_error", "timeout"):
            return last
        # oom / error -> try next ladder level
    return last


def run_graph(tid, graph, timeout):
    res = {"id": tid, "status": None, "wall_sec": None, "peak_vram_gb": None,
           "error": None, "output": None, "video": None,
           "disk_free_gb": round(disk_free_gb(), 1)}
    if res["disk_free_gb"] < 2.0:
        res["status"] = "skip_no_disk"
        return res

    mon = VramMonitor()
    mon.start()
    t0 = time.time()
    pid = None
    try:
        r = http_post("/prompt", {"prompt": graph, "client_id": "limit-test"})
        pid = r.get("prompt_id")
    except Exception as e:
        res["status"] = "submit_error"
        res["error"] = str(e)[:300]
        mon.stop_flag = True
        return res

    err = None
    while True:
        time.sleep(2)
        try:
            h = http_get(f"/history/{pid}")
        except Exception:
            h = {}
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") in ("success", "error"):
                err = st.get("messages")
                res["status"] = "ok" if st["status_str"] == "success" else "error"
                outputs = h[pid].get("outputs", {})
                files = []
                for node_out in outputs.values():
                    for img in node_out.get("images", []):
                        files.append(os.path.join(
                            OUT_DIR, img.get("subfolder", ""), img["filename"]))
                if files and res["status"] == "ok":
                    res["output"] = files[0]
                    try:
                        res["video"] = analyze_webp(files[0])
                    except Exception as e:
                        res["video"] = {"analyze_error": str(e)[:200]}
                break
        if time.time() - t0 > timeout:
            try:
                http_post("/interrupt", {})
            except Exception:
                pass
            for _ in range(15):
                time.sleep(2)
                try:
                    h = http_get(f"/history/{pid}")
                    if pid in h:
                        break
                except Exception:
                    pass
            res["status"] = "timeout"
            break

    res["wall_sec"] = round(time.time() - t0, 1)
    if mon.max_used is not None:
        res["peak_vram_gb"] = round(mon.max_used / 2**30, 2)
    mon.stop_flag = True

    if res["status"] == "error" and err:
        try:
            txt = json.dumps(err)
        except Exception:
            txt = str(err)
        low = txt.lower()
        if "out of memory" in low or "outofmemory" in txt:
            res["status"] = "oom"
        res["error"] = txt[:500]
    return res


def main():
    queue_file = sys.argv[1]
    tests = [json.loads(l) for l in open(queue_file) if l.strip()]
    with open(RESULTS, "a") as rf:
        for t in tests:
            print(f"RUN {t['id']}", flush=True)
            try:
                r = run_one(t)
            except Exception as e:
                r = {"id": t["id"], "status": "driver_error", "error": str(e)[:400]}
            rf.write(json.dumps(r, ensure_ascii=False) + "\n")
            rf.flush()
            print("DONE " + json.dumps(r, ensure_ascii=False)[:260], flush=True)


if __name__ == "__main__":
    main()
