"""轻量时域平滑节点（Temporal Smoothing Lite）

背景：ComfyUI-SA-Nodes-QQ 里的 TemporalSmoothingNode 因缺 omegaconf 等依赖整包导入失败，
本节点用同样的算法（时间维高斯加权平均，非空间模糊）独立实现，零外部依赖。
用途：接在 VAE Decode 之后、视频保存之前，抑制 AI 视频的时域闪烁/抖动。
"""
import torch


class TemporalSmoothingLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2,
                                        "tooltip": "时间窗口（奇数）。越大越平滑，过大动作会拖影"}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "混合强度：0=不处理，1=完全用平滑结果"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "smooth"
    CATEGORY = "image/postprocessing"

    def smooth(self, images, kernel_size=5, sigma=1.0, strength=1.0):
        B = images.shape[0]
        k = max(1, int(kernel_size) | 1)          # 强制奇数
        left = k // 2
        offs = torch.arange(-left, left + 1, dtype=torch.float32, device=images.device)
        w = torch.exp(-(offs ** 2) / (2.0 * max(sigma, 1e-3) ** 2))
        w = w / w.sum()
        x = images.to(torch.float32)
        out = torch.empty_like(x)
        idx = torch.arange(B, device=images.device)
        for j, o in enumerate(offs):
            nb = torch.clamp(idx + int(o), 0, B - 1)   # 边界钳制
            out += w[j] * x[nb]
        if strength < 1.0:
            out = x * (1.0 - strength) + out * strength
        return (out.clamp(0.0, 1.0).to(images.dtype),)


NODE_CLASS_MAPPINGS = {"TemporalSmoothingLite": TemporalSmoothingLite}
NODE_DISPLAY_NAME_MAPPINGS = {"TemporalSmoothingLite": "时域平滑（轻量）"}
