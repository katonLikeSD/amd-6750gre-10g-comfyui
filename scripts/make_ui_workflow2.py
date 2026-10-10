#!/usr/bin/env python3
"""生成带「I2V 起始图 + LoRA」模块的 UI 工作流（两个变体）
A) base35：基础 Q4_K_M + 35步/cfg5/dpmpp_2m_sde_heun+sgm_uniform
B) turbo ：Turbo Q5_K_S + 4步/cfg1/euler+simple
用法：python3 make_ui_workflow2.py [base35|turbo|both]
输出：ComfyUI/user/default/workflows/ 与仓库 ~/comfyui-amd6750gre-repo/workflows/
"""
import json, os, sys

LORA_DEFAULT = "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
START_IMG = "example.png"

POS_TEXT = ("A cinematic slow lateral tracking shot moving right: an orange cat walking slowly "
            "through sunlit green grass in a park at golden hour. A gentle breeze sways the "
            "grass and leaves; the cat's tail swings gently as it walks. Warm soft sunlight, "
            "slightly blurred background, steady smooth camera movement, clean composition, "
            "high quality, detailed.")
KEEP = ""
NEG = ("blurry, low quality, deformed, flickering, watermark, text, "
       "lens flare, bloom, overexposed, blown highlights, glow, hazy")


def note(variant):
    common = ("【怎么用】① 改「① 主提示词」（两段共用）→ ② 点运行\n"
              "【图生视频】「⓪ 起始图（I2V）」选中你的图 = 第一段从它开始；"
              "不想要就选中该节点按 Ctrl+B 旁路 → 回文生视频\n"
              "【LoRA】「⓪ LoRA」默认强度 0（不生效）；选文件并把强度调到 1.0 即生效。"
              "例：基础模型挂 Turbo LoRA（rank64, 1.0）也能 4 步出片（实测与官方 Turbo 几乎一致）\n"
              "【输出】output/video10s_ui/：seg1/seg2 无损 webp + short10s_241f mp4\n"
              "【注意】帧数必须 4n+1；241 帧全链尾段可能糊化，干净成片按锐度曲线裁尾+重定时")
    if variant == "turbo":
        return ("【Turbo 4 步版】4 步 / CFG 1 / euler / simple（模型卡推荐）；"
                "Wan2_2-TI2V-5B-Turbo-Q5_K_S.gguf，约 10-12 分钟出 241 帧\n" + common)
    return ("【基础 35 步版】35 步 / CFG 5 / dpmpp_2m_sde_heun + sgm_uniform；"
            "Wan2.2-TI2V-5B-Q4_K_M.gguf，约 40 分钟出 241 帧\n" + common)


VARIANTS = {
    "base35": dict(unet="Wan2.2-TI2V-5B-Q4_K_M.gguf",
                   ksw=[42, "fixed", 35, 5.0, "dpmpp_2m_sde_heun", "sgm_uniform", 1.0],
                   comfy_name="10s_链式两段_可改文本.json",
                   repo_name="wan22_10s_chain_241f.json"),
    "turbo": dict(unet="Wan2_2-TI2V-5B-Turbo-Q5_K_S.gguf",
                  ksw=[42, "fixed", 4, 1.0, "euler", "simple", 1.0],
                  comfy_name="10s_链式_Turbo4步.json",
                  repo_name="wan22_10s_turbo4step.json"),
}


def build(variant):
    v = VARIANTS[variant]
    nodes, links, _lid = [], [], [0]

    def add(nid, ntype, pos, size, widgets=None, inputs=None, outputs=None, title=None,
            properties=None):
        n = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {},
             "order": len(nodes), "mode": 0, "inputs": inputs or [], "outputs": outputs or [],
             "widgets_values": widgets if widgets is not None else [],
             "properties": properties or {"Node name for S&R": ntype}}
        if title:
            n["title"] = title
        nodes.append(n)
        return n

    def out(n, name, type_):
        n["outputs"].append({"name": name, "type": type_, "links": [],
                             "slot_index": len(n["outputs"])})
        return len(n["outputs"]) - 1

    def link(src, src_slot, dst, dst_slot, type_, widget_name=None):
        _lid[0] += 1
        i = _lid[0]
        links.append([i, src["id"], src_slot, dst["id"], dst_slot, type_])
        src["outputs"][src_slot]["links"].append(i)
        e = {"name": dst["_in_names"][dst_slot], "type": type_, "link": i}
        if widget_name:
            e["widget"] = {"name": widget_name}
        dst["inputs"].append(e)

    def iname(n, *names):
        n["_in_names"] = list(names)

    # 加载器
    n_vae = add(1, "VAELoader", [40, 60], [280, 60], ["wan2.2_vae.safetensors"])
    out(n_vae, "VAE", "VAE")
    n_clip = add(2, "CLIPLoader", [40, 160], [300, 100],
                 ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "cpu"])
    out(n_clip, "CLIP", "CLIP")
    n_unet = add(3, "UnetLoaderGGUF", [40, 300], [300, 60], [v["unet"]])
    out(n_unet, "MODEL", "MODEL")

    # 起始图（I2V 可选）+ LoRA（默认强度 0）
    n_img = add(25, "LoadImage", [40, 400], [320, 300], [START_IMG, START_IMG],
                title="⓪ 起始图（I2V；Ctrl+B 旁路=文生视频）")
    out(n_img, "IMAGE", "IMAGE")
    n_lora = add(24, "LoraLoaderModelOnly", [40, 740], [320, 110],
                 [LORA_DEFAULT, 0.0],
                 title="⓪ LoRA（默认强度 0=不生效；选文件+调强度启用）")
    iname(n_lora, "model")
    out(n_lora, "MODEL", "MODEL")

    # 文本
    n_txt = add(4, "PrimitiveString", [400, 60], [560, 150], [POS_TEXT],
                title="① 主提示词（改这里，两段共用）")
    out(n_txt, "STRING", "STRING")
    n_cat = add(5, "StringConcatenate", [400, 250], [560, 110], ["", KEEP, " "],
                title="② 第二段附加句（一般不用改）")
    iname(n_cat, "string_a", "string_b", "delimiter")
    out(n_cat, "STRING", "STRING")
    n_pos1 = add(6, "CLIPTextEncode", [1000, 60], [340, 140], [""], title="第一段正向")
    iname(n_pos1, "clip", "text")
    out(n_pos1, "CONDITIONING", "CONDITIONING")
    n_neg = add(7, "CLIPTextEncode", [1000, 240], [340, 140], [NEG], title="负向")
    iname(n_neg, "clip")
    out(n_neg, "CONDITIONING", "CONDITIONING")
    n_pos2 = add(8, "CLIPTextEncode", [1000, 420], [340, 140], [""],
                 title="第二段正向（=①+②自动拼接）")
    iname(n_pos2, "clip", "text")
    out(n_pos2, "CONDITIONING", "CONDITIONING")

    link(n_clip, 0, n_pos1, 0, "CLIP")
    link(n_txt, 0, n_pos1, 1, "STRING", widget_name="text")
    link(n_clip, 0, n_neg, 0, "CLIP")
    link(n_clip, 0, n_pos2, 0, "CLIP")
    link(n_txt, 0, n_cat, 0, "STRING", widget_name="string_a")
    link(n_cat, 0, n_pos2, 1, "STRING", widget_name="text")
    link(n_unet, 0, n_lora, 0, "MODEL")

    # 第一段
    n_lv1 = add(9, "Wan22ImageToVideoLatent", [1380, 60], [300, 130], [704, 480, 121, 1],
                title="第一段 latent（帧数须 4n+1）")
    iname(n_lv1, "vae", "start_image")
    out(n_lv1, "LATENT", "LATENT")
    n_ks1 = add(10, "KSampler", [1720, 60], [300, 260], list(v["ksw"]), title="第一段采样")
    iname(n_ks1, "model", "positive", "negative", "latent_image")
    out(n_ks1, "LATENT", "LATENT")
    n_vd1 = add(11, "VAEDecodeTiled", [2060, 60], [300, 130], [128, 32, 8, 4],
                title="第一段解码（128/32/8/4 别改）")
    iname(n_vd1, "samples", "vae")
    out(n_vd1, "IMAGE", "IMAGE")
    link(n_vae, 0, n_lv1, 0, "VAE")
    link(n_img, 0, n_lv1, 1, "IMAGE")
    link(n_lora, 0, n_ks1, 0, "MODEL")
    link(n_pos1, 0, n_ks1, 1, "CONDITIONING")
    link(n_neg, 0, n_ks1, 2, "CONDITIONING")
    link(n_lv1, 0, n_ks1, 3, "LATENT")
    link(n_ks1, 0, n_vd1, 0, "LATENT")
    link(n_vae, 0, n_vd1, 1, "VAE")

    # 尾帧
    n_last = add(12, "ImageFromBatch", [2400, 60], [300, 100], [120, 1], title="取第一段尾帧")
    iname(n_last, "image")
    out(n_last, "IMAGE", "IMAGE")
    link(n_vd1, 0, n_last, 0, "IMAGE")

    # 第二段
    n_lv2 = add(13, "Wan22ImageToVideoLatent", [1380, 300], [300, 130], [704, 480, 121, 1],
                title="第二段 latent（start_image=尾帧）")
    iname(n_lv2, "vae", "start_image")
    out(n_lv2, "LATENT", "LATENT")
    n_ks2 = add(14, "KSampler", [1720, 300], [300, 260], list(v["ksw"]), title="第二段采样")
    iname(n_ks2, "model", "positive", "negative", "latent_image")
    out(n_ks2, "LATENT", "LATENT")
    n_vd2 = add(15, "VAEDecodeTiled", [2060, 300], [300, 130], [128, 32, 8, 4], title="第二段解码")
    iname(n_vd2, "samples", "vae")
    out(n_vd2, "IMAGE", "IMAGE")
    link(n_vae, 0, n_lv2, 0, "VAE")
    link(n_last, 0, n_lv2, 1, "IMAGE")
    link(n_lora, 0, n_ks2, 0, "MODEL")
    link(n_pos2, 0, n_ks2, 1, "CONDITIONING")
    link(n_neg, 0, n_ks2, 2, "CONDITIONING")
    link(n_lv2, 0, n_ks2, 3, "LATENT")
    link(n_ks2, 0, n_vd2, 0, "LATENT")
    link(n_vae, 0, n_vd2, 1, "VAE")

    # 保存 + 拼接出片
    n_sw1 = add(16, "SaveAnimatedWEBP", [2400, 240], [300, 130],
                ["video10s_ui/seg1", 24.0, True, 100, "default"], title="第一段无损 webp")
    iname(n_sw1, "images")
    out(n_sw1, "IMAGE", "IMAGE")
    link(n_vd1, 0, n_sw1, 0, "IMAGE")
    n_sw2 = add(17, "SaveAnimatedWEBP", [2400, 420], [300, 130],
                ["video10s_ui/seg2", 24.0, True, 100, "default"], title="第二段无损 webp")
    iname(n_sw2, "images")
    out(n_sw2, "IMAGE", "IMAGE")
    link(n_vd2, 0, n_sw2, 0, "IMAGE")
    n_full1 = add(18, "ImageFromBatch", [2740, 60], [280, 100], [0, 121], title="第一段全部 121 帧")
    iname(n_full1, "image")
    out(n_full1, "IMAGE", "IMAGE")
    link(n_vd1, 0, n_full1, 0, "IMAGE")
    n_full2 = add(19, "ImageFromBatch", [2740, 200], [280, 100], [1, 120],
                  title="第二段去掉共享首帧，取 120 帧")
    iname(n_full2, "image")
    out(n_full2, "IMAGE", "IMAGE")
    link(n_vd2, 0, n_full2, 0, "IMAGE")
    n_bat = add(20, "BatchImagesNode", [3080, 60], [260, 170], [], title="拼成 241 帧 = 10.04s@24fps")
    iname(n_bat, "images.image0", "images.image1", "images.image2")
    out(n_bat, "IMAGE", "IMAGE")
    link(n_full1, 0, n_bat, 0, "IMAGE")
    link(n_full2, 0, n_bat, 1, "IMAGE")
    n_bat["inputs"].append({"label": "image2", "name": "images.image2", "shape": 7,
                            "type": "IMAGE", "link": None})
    n_cv = add(21, "CreateVideo", [3400, 60], [260, 100], [24], title="24 fps")
    iname(n_cv, "images", "audio")
    out(n_cv, "VIDEO", "VIDEO")
    link(n_bat, 0, n_cv, 0, "IMAGE")
    n_cv["inputs"].append({"name": "audio", "shape": 7, "type": "AUDIO", "link": None})
    n_sv = add(22, "SaveVideo", [3700, 60], [340, 130],
               ["video10s_ui/short10s_241f", "auto", "auto"], title="成片 mp4")
    iname(n_sv, "video")
    out(n_sv, "VIDEO", "VIDEO")
    link(n_cv, 0, n_sv, 0, "VIDEO")
    n_note = add(23, "Note", [400, 480], [900, 320], [note(variant)], title="说明 / 使用指引",
                 properties={})
    n_note["color"] = "#222"
    n_note["bgcolor"] = "#000"

    for n in nodes:
        n.pop("_in_names", None)
    return {"last_node_id": max(n["id"] for n in nodes), "last_link_id": _lid[0],
            "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {},
            "version": 0.4}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    targets = ["base35", "turbo"] if which == "both" else [which]
    repo = os.path.expanduser("~/comfyui-amd6750gre-repo/workflows")
    comfy = os.path.expanduser("~/ComfyUI/user/default/workflows")
    for v in targets:
        wf = build(v)
        for d, name in ((comfy, VARIANTS[v]["comfy_name"]), (repo, VARIANTS[v]["repo_name"])):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(wf, f, ensure_ascii=False, indent=1)
            print("written:", os.path.join(d, name), f"({len(wf['nodes'])} nodes, {len(wf['links'])} links)")