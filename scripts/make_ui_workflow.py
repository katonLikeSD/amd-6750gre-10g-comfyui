#!/usr/bin/env python3
"""生成 ComfyUI 网页界面(UI)格式工作流：10s 两段链式、可改文本、一键出 mp4。
输出到 ComfyUI/user/default/workflows/ 与 ~/video10s/。
"""
import json, os

OUT_DIRS = [os.path.expanduser("~/ComfyUI/user/default/workflows"),
            os.path.expanduser("~/video10s")]
NAME = "10s_链式两段_可改文本.json"

POS_TEXT = ("A cinematic slow lateral tracking shot moving right: an orange cat walking slowly "
            "through sunlit green grass in a park at golden hour. A gentle breeze sways the "
            "grass and leaves; the cat's tail swings gently as it walks. Warm soft sunlight, "
            "slightly blurred background, steady smooth camera movement, clean composition, "
            "high quality, detailed.")
KEEP = ("")   # DS 复审结论：seg2 直接用与 seg1 完全相同的提示词，不要加"保持/上一帧"类句子
NEG = "blurry, low quality, deformed, flickering, watermark, text"
NOTE = ("【怎么用】① 改「① 主提示词」里的文字（两段共用）→ ② 点底部 Queue 运行\n"
        "【耗时】约 40 分钟（10G 卡：121帧×35步×704×480 ×2 段 + 瓦片解码）\n"
        "【输出】ComfyUI/output/video10s_ui/ ：seg1/seg2 无损 webp + short10s_241f mp4\n"
        "【想改别的】步数在 KSampler 的 steps；分辨率/帧数在 Wan22ImageToVideoLatent（帧数必须 4n+1，如 121）\n"
        "【实测要点】①帧数别低于 33——Wan2.2-5B 超短长度会退化成彩色噪声（17 帧必崩）\n"
        "② 第二段默认与第一段同文本：扩散模型看不到\"上一帧\"，连贯性靠 start_image，不靠文字\n"
        "③ 镜头方向要写死（如 moving right），两段一致，否则接缝处容易方向漂移")

nodes = []
links = []
_lid = [0]


def add(nid, ntype, pos, size, widgets=None, inputs=None, outputs=None, title=None,
        properties=None):
    n = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {},
         "order": len(nodes), "mode": 0,
         "inputs": inputs or [], "outputs": outputs or [],
         "widgets_values": widgets if widgets is not None else [],
         "properties": properties or {"Node name for S&R": ntype}}
    if title:
        n["title"] = title
    nodes.append(n)
    return n


def out(n, name, type_):
    n["outputs"].append({"name": name, "type": type_, "links": [], "slot_index": len(n["outputs"])})
    return len(n["outputs"]) - 1


def link(src, src_slot, dst, dst_slot, type_, widget_name=None):
    _lid[0] += 1
    i = _lid[0]
    links.append([i, src["id"], src_slot, dst["id"], dst_slot, type_])
    src["outputs"][src_slot]["links"].append(i)
    entry = {"name": dst["_in_names"][dst_slot], "type": type_, "link": i}
    if widget_name:
        entry["widget"] = {"name": widget_name}
    dst["inputs"].append(entry)
    return i


def iname(n, *names):
    n["_in_names"] = list(names)


# ---- 加载器 ----
n_vae = add(1, "VAELoader", [40, 60], [280, 60], ["wan2.2_vae.safetensors"])
out(n_vae, "VAE", "VAE")
n_clip = add(2, "CLIPLoader", [40, 160], [300, 100],
             ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "cpu"])
out(n_clip, "CLIP", "CLIP")
n_unet = add(3, "UnetLoaderGGUF", [40, 300], [300, 60], ["Wan2.2-TI2V-5B-Q4_K_M.gguf"])
out(n_unet, "MODEL", "MODEL")

# ---- 文本（唯一需要改的地方）----
n_txt = add(4, "PrimitiveString", [40, 420], [560, 150], [POS_TEXT],
            title="① 主提示词（改这里，两段共用）")
out(n_txt, "STRING", "STRING")
n_cat = add(5, "StringConcatenate", [40, 620], [560, 110], ["", KEEP, " "],
            title="② 第二段附加句（一般不用改）")
iname(n_cat, "string_a", "string_b", "delimiter")
out(n_cat, "STRING", "STRING")

n_pos1 = add(6, "CLIPTextEncode", [700, 60], [340, 140], [""], title="第一段正向")
iname(n_pos1, "clip", "text")
out(n_pos1, "CONDITIONING", "CONDITIONING")
n_neg = add(7, "CLIPTextEncode", [700, 240], [340, 140], [NEG], title="负向")
iname(n_neg, "clip")
out(n_neg, "CONDITIONING", "CONDITIONING")
n_pos2 = add(8, "CLIPTextEncode", [700, 420], [340, 140], [""], title="第二段正向（=①+②自动拼接）")
iname(n_pos2, "clip", "text")
out(n_pos2, "CONDITIONING", "CONDITIONING")

link(n_clip, 0, n_pos1, 0, "CLIP")
link(n_txt, 0, n_pos1, 1, "STRING", widget_name="text")
link(n_clip, 0, n_neg, 0, "CLIP")
link(n_clip, 0, n_pos2, 0, "CLIP")
link(n_txt, 0, n_cat, 0, "STRING", widget_name="string_a")
link(n_cat, 0, n_pos2, 1, "STRING", widget_name="text")

# ---- 第一段：T2V 121帧 ----
n_lv1 = add(9, "Wan22ImageToVideoLatent", [1100, 60], [300, 130], [704, 480, 121, 1],
            title="第一段 latent（帧数须 4n+1）")
iname(n_lv1, "vae")
out(n_lv1, "LATENT", "LATENT")
n_ks1 = add(10, "KSampler", [1440, 60], [300, 260], [42, "fixed", 35, 5.0,
            "dpmpp_2m_sde_heun", "sgm_uniform", 1.0], title="第一段采样")
iname(n_ks1, "model", "positive", "negative", "latent_image")
out(n_ks1, "LATENT", "LATENT")
n_vd1 = add(11, "VAEDecodeTiled", [1780, 60], [300, 130], [128, 32, 8, 4],
            title="第一段解码（128/32/8/4 别改）")
iname(n_vd1, "samples", "vae")
out(n_vd1, "IMAGE", "IMAGE")

link(n_vae, 0, n_lv1, 0, "VAE")
link(n_unet, 0, n_ks1, 0, "MODEL")
link(n_pos1, 0, n_ks1, 1, "CONDITIONING")
link(n_neg, 0, n_ks1, 2, "CONDITIONING")
link(n_lv1, 0, n_ks1, 3, "LATENT")
link(n_ks1, 0, n_vd1, 0, "LATENT")
link(n_vae, 0, n_vd1, 1, "VAE")

# ---- 衔接：取第一段尾帧 ----
n_last = add(12, "ImageFromBatch", [2120, 60], [300, 100], [120, 1], title="取第一段尾帧")
iname(n_last, "image")
out(n_last, "IMAGE", "IMAGE")
link(n_vd1, 0, n_last, 0, "IMAGE")

# ---- 第二段：I2V 链式 ----
n_lv2 = add(13, "Wan22ImageToVideoLatent", [1100, 300], [300, 130], [704, 480, 121, 1],
            title="第二段 latent（start_image=尾帧）")
iname(n_lv2, "vae", "start_image")
out(n_lv2, "LATENT", "LATENT")
n_ks2 = add(14, "KSampler", [1440, 300], [300, 260], [42, "fixed", 35, 5.0,
            "dpmpp_2m_sde_heun", "sgm_uniform", 1.0], title="第二段采样")
iname(n_ks2, "model", "positive", "negative", "latent_image")
out(n_ks2, "LATENT", "LATENT")
n_vd2 = add(15, "VAEDecodeTiled", [1780, 300], [300, 130], [128, 32, 8, 4], title="第二段解码")
iname(n_vd2, "samples", "vae")
out(n_vd2, "IMAGE", "IMAGE")

link(n_vae, 0, n_lv2, 0, "VAE")
link(n_last, 0, n_lv2, 1, "IMAGE")
link(n_unet, 0, n_ks2, 0, "MODEL")
link(n_pos2, 0, n_ks2, 1, "CONDITIONING")
link(n_neg, 0, n_ks2, 2, "CONDITIONING")
link(n_lv2, 0, n_ks2, 3, "LATENT")
link(n_ks2, 0, n_vd2, 0, "LATENT")
link(n_vae, 0, n_vd2, 1, "VAE")

# ---- 保存两段 + 拼接成片 ----
n_sw1 = add(16, "SaveAnimatedWEBP", [2120, 240], [300, 130],
            ["video10s_ui/seg1", 24.0, True, 100, "default"], title="第一段无损 webp")
iname(n_sw1, "images")
out(n_sw1, "IMAGE", "IMAGE")
link(n_vd1, 0, n_sw1, 0, "IMAGE")

n_sw2 = add(17, "SaveAnimatedWEBP", [2120, 420], [300, 130],
            ["video10s_ui/seg2", 24.0, True, 100, "default"], title="第二段无损 webp")
iname(n_sw2, "images")
out(n_sw2, "IMAGE", "IMAGE")
link(n_vd2, 0, n_sw2, 0, "IMAGE")

n_full1 = add(18, "ImageFromBatch", [2460, 60], [280, 100], [0, 121], title="第一段全部 121 帧")
iname(n_full1, "image")
out(n_full1, "IMAGE", "IMAGE")
link(n_vd1, 0, n_full1, 0, "IMAGE")

n_full2 = add(19, "ImageFromBatch", [2460, 200], [280, 100], [1, 120],
              title="第二段去掉共享首帧，取 120 帧")
iname(n_full2, "image")
out(n_full2, "IMAGE", "IMAGE")
link(n_vd2, 0, n_full2, 0, "IMAGE")

n_bat = add(20, "BatchImagesNode", [2800, 60], [260, 170], [], title="拼成 241 帧 = 10.04s@24fps")
iname(n_bat, "images.image0", "images.image1", "images.image2")
out(n_bat, "IMAGE", "IMAGE")
link(n_full1, 0, n_bat, 0, "IMAGE")
link(n_full2, 0, n_bat, 1, "IMAGE")
n_bat["inputs"].append({"label": "image2", "name": "images.image2", "shape": 7,
                        "type": "IMAGE", "link": None})

n_cv = add(21, "CreateVideo", [3120, 60], [260, 100], [24], title="24 fps")
iname(n_cv, "images", "audio")
out(n_cv, "VIDEO", "VIDEO")
link(n_bat, 0, n_cv, 0, "IMAGE")
n_cv["inputs"].append({"name": "audio", "shape": 7, "type": "AUDIO", "link": None})

n_sv = add(22, "SaveVideo", [3420, 60], [340, 130],
           ["video10s_ui/short10s_241f", "auto", "auto"], title="成片 mp4")
iname(n_sv, "video")
out(n_sv, "VIDEO", "VIDEO")
link(n_cv, 0, n_sv, 0, "VIDEO")

n_note = add(23, "Note", [40, 780], [620, 260], [NOTE], title="说明 / 使用指引",
             properties={})
n_note["color"] = "#222"
n_note["bgcolor"] = "#000"

for n in nodes:
    n.pop("_in_names", None)

wf = {"last_node_id": max(n["id"] for n in nodes), "last_link_id": _lid[0],
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {},
      "version": 0.4}

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, NAME), "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=1)
    print("written:", os.path.join(d, NAME))
print("nodes:", len(nodes), "links:", len(links))