#!/usr/bin/env python3
"""10s 短片生成进度面板：python3 progress.py [--once]"""
import json, os, re, sys, time, urllib.request

BASE = "http://127.0.0.1:8188"
LOG = "/tmp/comfy10s.log"
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
STATE = os.path.expanduser("~/video10s/state.json")


def comfy_tail():
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 5000))
            tail = f.read().decode("utf-8", "ignore")
        parts = [p for p in re.split(r"[\r\n]", tail) if p.strip()]
        for p in reversed(parts):
            if re.search(r"\d+%\|", p) or "s/it" in p or "it/s" in p:
                return p.strip()[:110]
        return parts[-1].strip()[:110] if parts else "(无)"
    except Exception as e:
        return "(日志读取失败: %s)" % e


def stage():
    try:
        q = json.loads(urllib.request.urlopen(BASE + "/queue", timeout=5).read())
    except Exception:
        return "ComfyUI 未响应（是不是已经关了？）", 0
    run, pend = q.get("queue_running") or [], q.get("queue_pending") or []
    if not run:
        return ("队列空闲" if not pend else f"待运行 {len(pend)} 个任务"), len(pend)
    prompt = run[0][2]
    tag, steps = "?", "?"
    for v in prompt.values():
        if v.get("class_type") == "SaveAnimatedWEBP":
            tag = v["inputs"].get("filename_prefix", "?").split("/")[-1]
        if v.get("class_type") == "KSampler":
            steps = v["inputs"].get("steps")
    return f"正在生成 {tag}（{steps} 步）", len(pend)


def vram():
    try:
        with open(VRAM) as f:
            return int(f.read().strip()) / 2**30
    except Exception:
        return None


def done_flags():
    try:
        s = json.load(open(STATE))
    except Exception:
        return ""
    f = []
    for k in ("draft", "seg1", "seg2"):
        if k in s:
            f.append(f"{k}=OK")
    if os.path.exists(os.path.expanduser("~/video10s/seam_report.json")):
        f.append("拼接=OK")
    return " ".join(f)


def draw():
    st, _ = stage()
    v = vram()
    lines = [
        "=== 10s 短片生成进度 ===",
        f"阶段  : {st}",
        f"采样  : {comfy_tail()}",
        f"显存  : {v:.2f} G (10G 卡)" if v else "显存  : n/a",
        f"已完成: {done_flags() or '(无)'}",
        "",
        "（Ctrl+C 退出；成片将出现在 ~/video10s/）",
    ]
    sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if "--once" in sys.argv:
        st, _ = stage()
        v = vram()
        print("阶段  :", st)
        print("采样  :", comfy_tail())
        print("显存  :", f"{v:.2f} G" if v else "n/a")
        print("已完成:", done_flags() or "(无)")
    else:
        while True:
            draw()
            time.sleep(2)