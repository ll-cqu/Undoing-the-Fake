import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from undo_the_fake.configs.attributes import ALPHA_LEVEL_TO_MIDPOINT, ATTRIBUTE_CONFIG, BETA_DEFAULT
from undo_the_fake.configs.runtime import E4E_CKPT, E4E_ROOT, FACE_EDITING_ROOT, FS3_PATH, STYLECLIP_ROOT, STYLEGAN_PKL


@dataclass(frozen=True)
class GanPaths:
    styleclip_root: str = STYLECLIP_ROOT
    e4e_root: str = E4E_ROOT
    stylegan_pkl: str = STYLEGAN_PKL
    fs3_path: str = FS3_PATH
    e4e_ckpt: str = E4E_CKPT


IMG_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

LPIPS_TRANSFORM = IMG_TRANSFORM
_GAN_CACHE = {}


def _arcface_utils():
    if FACE_EDITING_ROOT not in sys.path:
        sys.path.insert(0, FACE_EDITING_ROOT)
    from arcface_utils import get_arcface, arcface_sim
    return get_arcface, arcface_sim


def get_arcface(device: str):
    fn, _ = _arcface_utils()
    return fn(device)


def arcface_sim(a, b, device: str):
    _, fn = _arcface_utils()
    return fn(a, b, device)


def _with_styleclip_path(paths: GanPaths):
    extra = [paths.styleclip_root, paths.e4e_root, os.path.join(paths.styleclip_root, "global_torch")]
    original = sys.path.copy()
    for item in extra:
        if item not in sys.path:
            sys.path.insert(0, item)
    return original


def load_e4e(device: str = "cuda", ckpt_path: str = E4E_CKPT):
    from argparse import Namespace
    from models.psp import pSp

    ckpt = torch.load(ckpt_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = ckpt_path
    opts["device"] = device
    return pSp(Namespace(**opts)).eval().to(device)


def load_styleclip_global(device: str = "cuda", stylegan_pkl: str = STYLEGAN_PKL, fs3_path: str = FS3_PATH):
    import clip
    from manipulate import Manipulator, LoadModel

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    manipulator = Manipulator(dataset_name="ffhq")
    manipulator.device = device
    manipulator.G = LoadModel(stylegan_pkl, device)
    manipulator.SetGParameters()
    manipulator.GenerateS(num_img=10000)
    manipulator.GetCodeMS()
    return clip_model, manipulator, np.load(fs3_path)


def precompute_directions(clip_model, manipulator, fs3):
    from StyleCLIP import GetDt, GetBoundary

    directions = {}
    for attr, cfg in ATTRIBUTE_CONFIG.items():
        beta = cfg.get("beta_override", BETA_DEFAULT)
        dt = GetDt([cfg["target"], cfg["neutral"]], clip_model)
        boundary, num_ch = GetBoundary(fs3, dt, manipulator, threshold=beta)
        directions[attr] = boundary
        print(f"  [direction] {attr:<12} beta={beta}  num_ch={num_ch}", flush=True)
    return directions


def get_gan_models(device: str, paths: GanPaths = GanPaths(), with_lpips: bool = False):
    key = (device, paths, with_lpips)
    if key in _GAN_CACHE:
        return _GAN_CACHE[key]

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    import time

    original = _with_styleclip_path(paths)
    try:
        t0 = time.time()
        e4e = load_e4e(device=device, ckpt_path=paths.e4e_ckpt)
        clip_model, manipulator, fs3 = load_styleclip_global(
            device=device,
            stylegan_pkl=paths.stylegan_pkl,
            fs3_path=paths.fs3_path,
        )
        directions = precompute_directions(clip_model, manipulator, fs3)
        get_arcface(device)

        lpips_fn = None
        if with_lpips:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            lpips_fn.eval()

        print(f"[GAN {device}] init done in {time.time() - t0:.1f}s", flush=True)
        _GAN_CACHE[key] = (e4e, manipulator, directions, lpips_fn)
    finally:
        sys.path = original

    return _GAN_CACHE[key]


@torch.no_grad()
def encode(e4e, img: Image.Image, device: str):
    tensor = IMG_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    _, w = e4e(tensor, randomize_noise=False, return_latents=True)
    return w


@torch.no_grad()
def encode_batch(e4e, img_paths: list, device: str):
    tensors, valid = [], []
    for path in img_paths:
        try:
            tensors.append(IMG_TRANSFORM(Image.open(path).convert("RGB")))
            valid.append(path)
        except Exception:
            pass
    if not tensors:
        return [], []
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        _, w = e4e(batch, randomize_noise=False, return_latents=True)
    return valid, [w[i:i + 1] for i in range(len(tensors))]


@torch.no_grad()
def get_s_codes(manipulator, w_plus):
    return manipulator.S2List(manipulator.G.synthesis.W2S(w_plus))


def apply_boundary(manipulator, s_codes, boundary, alpha: float):
    manipulator.num_images = 1
    manipulator.alpha = [alpha]
    manipulator.step = 1
    codes = manipulator.MSCode(s_codes, boundary)
    return [c[:, 0, :] for c in codes]


@torch.no_grad()
def generate_from_s(manipulator, s_codes) -> Image.Image:
    manipulator.num_images = 1
    manipulator.alpha = [0]
    manipulator.step = 1
    codes = manipulator.MSCode(s_codes, [np.zeros_like(b) for b in s_codes])
    return Image.fromarray(manipulator.GenerateImg(codes)[0, 0])


@torch.no_grad()
def lpips_score(lpips_fn, a: Image.Image, b: Image.Image, device: str) -> float:
    ta = LPIPS_TRANSFORM(a.convert("RGB")).unsqueeze(0).to(device)
    tb = LPIPS_TRANSFORM(b.convert("RGB")).unsqueeze(0).to(device)
    return float(lpips_fn(ta, tb).item())


def run_recovery(
    edited_img: Image.Image,
    recon_img: Image.Image,
    pred_manipulations: list,
    device: str,
    use_alpha_abs: bool = False,
    with_lpips: bool = False,
):
    try:
        e4e, manipulator, directions, lpips_fn = get_gan_models(device, with_lpips=with_lpips)
        s_codes = get_s_codes(manipulator, encode(e4e, edited_img, device))
        for manip in pred_manipulations:
            attr = manip.get("attr")
            action = manip.get("action")
            level = manip.get("alpha_level")
            if attr not in directions or not action:
                continue
            if use_alpha_abs and manip.get("alpha_abs") is not None:
                alpha_abs = float(manip["alpha_abs"])
            else:
                alpha_abs = ALPHA_LEVEL_TO_MIDPOINT.get(level, 1.0)
            recovery_alpha = -alpha_abs if action == "add" else +alpha_abs
            s_codes = apply_boundary(manipulator, s_codes, directions[attr], recovery_alpha)
        recovered = generate_from_s(manipulator, s_codes)
        lp_rec = lpips_score(lpips_fn, recovered, recon_img, device) if with_lpips and lpips_fn is not None else None
        return recovered, lp_rec
    except Exception as exc:
        print(f"[recovery] {exc}", flush=True)
        return None, None


def run_arc_recovery(edited_img: Image.Image, recon_img: Image.Image, pred_manipulations: list, device: str):
    recovered, _ = run_recovery(edited_img, recon_img, pred_manipulations, device, with_lpips=False)
    if recovered is None:
        return None, edited_img
    return arcface_sim(recon_img, recovered, device), recovered
