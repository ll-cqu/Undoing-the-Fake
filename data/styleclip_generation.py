import json
import os
import sys
import traceback
from collections import defaultdict
from queue import Queue as ThreadQueue
from threading import Thread

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from undo_the_fake.configs.attributes import ATTRIBUTE_CONFIG, BETA_DEFAULT
from undo_the_fake.configs.runtime import E4E_ROOT, FS3_PATH, STYLECLIP_ROOT, STYLEGAN_PKL
from undo_the_fake.data.plan_sampler import sample_alpha


for path in [STYLECLIP_ROOT, E4E_ROOT, os.path.join(STYLECLIP_ROOT, "global_torch")]:
    if path not in sys.path:
        sys.path.insert(0, path)


IMG_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def load_e4e(ckpt_path: str = "face_editing/pretrained/e4e_ffhq_encode.pt", device: str = "cuda"):
    from argparse import Namespace
    from models.psp import pSp

    ckpt = torch.load(ckpt_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = ckpt_path
    opts["device"] = device
    return pSp(Namespace(**opts)).eval().to(device)


def load_styleclip_global(
    device: str = "cuda",
    stylegan_pkl: str = STYLEGAN_PKL,
    fs3_path: str = FS3_PATH,
):
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


def precompute_all_directions(clip_model, manipulator, fs3) -> dict:
    from StyleCLIP import GetDt, GetBoundary

    direction_cache = {}
    for attr, cfg in ATTRIBUTE_CONFIG.items():
        beta = cfg.get("beta_override", BETA_DEFAULT)
        cache_key = (attr, beta)
        if cache_key in direction_cache:
            continue
        dt = GetDt([cfg["target"], cfg["neutral"]], clip_model)
        boundary, num_ch = GetBoundary(fs3, dt, manipulator, threshold=beta)
        direction_cache[cache_key] = (boundary, int(num_ch))
        print(f"  [direction] {attr:<12} beta={beta}  num_ch={num_ch}")
    print(f"[direction] cached {len(direction_cache)} directions")
    return direction_cache


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


def get_s_codes(manipulator, w_plus):
    with torch.no_grad():
        return manipulator.S2List(manipulator.G.synthesis.W2S(w_plus))


def apply_edit(manipulator, s_codes, boundary, alpha: float):
    manipulator.num_images = 1
    manipulator.alpha = [alpha]
    manipulator.step = 1
    codes = manipulator.MSCode(s_codes, boundary)
    return [code[:, 0, :] for code in codes]


def generate_from_s(manipulator, s_codes) -> Image.Image:
    manipulator.num_images = 1
    manipulator.alpha = [0]
    manipulator.step = 1
    codes = manipulator.MSCode(s_codes, [np.zeros_like(boundary) for boundary in s_codes])
    return Image.fromarray(manipulator.GenerateImg(codes)[0, 0])


class AsyncSaver:
    def __init__(self, num_threads: int = 4):
        self.queue = ThreadQueue(maxsize=256)
        self.threads = [Thread(target=self._worker, daemon=True) for _ in range(num_threads)]
        for thread in self.threads:
            thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            img, path = item
            try:
                img.save(path)
            except Exception as exc:
                print(f"Save failed {path}: {exc}")

    def save(self, img: Image.Image, path: str):
        self.queue.put((img, path))

    def stop(self):
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()


def execute_plan(e4e, manipulator, direction_cache: dict, img_path: str, w_plus, edit_plan: list, output_dir: str, variant: int, saver: AsyncSaver, lpips_fn=None) -> dict:
    img_stem = os.path.splitext(os.path.basename(img_path))[0]
    var_stem = f"{img_stem}_v{variant}"
    os.makedirs(output_dir, exist_ok=True)

    w_plus_path = os.path.join(output_dir, f"{img_stem}_wplus.npy")
    if not os.path.exists(w_plus_path):
        np.save(w_plus_path, w_plus.cpu().numpy())

    recon_path = os.path.join(output_dir, f"{img_stem}_recon.png")
    s_recon = get_s_codes(manipulator, w_plus)
    if not os.path.exists(recon_path):
        saver.save(generate_from_s(manipulator, s_recon), recon_path)

    s_current = [code.clone() if hasattr(code, "clone") else np.copy(code) for code in s_recon]
    edit_steps = []

    for edit in edit_plan:
        attr = edit["attr"]
        action = edit["action"]
        cfg = ATTRIBUTE_CONFIG[attr]
        beta = cfg.get("beta_override", BETA_DEFAULT)
        boundary, num_ch = direction_cache[(attr, beta)]
        alpha_level, alpha_abs = sample_alpha(attr)
        alpha_signed = alpha_abs if action == "add" else -alpha_abs

        s_current = apply_edit(manipulator, s_current, boundary, alpha_signed)
        edit_steps.append({
            "attr": attr,
            "action": action,
            "alpha_level": alpha_level,
            "alpha_abs": alpha_abs,
            "alpha_signed": alpha_signed,
            "beta": beta,
            "num_channels": num_ch,
            "celeba_attr": edit.get("celeba_attr"),
            "celeba_value": edit.get("celeba_value"),
        })

    attr_tag = "_".join(f"{edit['attr']}_{edit['action'][0]}" for edit in edit_steps)
    edited_path = os.path.join(output_dir, f"{var_stem}_{attr_tag}.png")
    edited_img = generate_from_s(manipulator, s_current)
    saver.save(edited_img, edited_path)

    return {
        "image": os.path.basename(img_path),
        "variant": variant,
        "original_path": os.path.abspath(img_path),
        "recon_path": os.path.abspath(recon_path),
        "edited_path": os.path.abspath(edited_path),
        "w_plus_path": os.path.abspath(w_plus_path),
        "num_edits": len(edit_steps),
        "attrs": [edit["attr"] for edit in edit_steps],
        "actions": [edit["action"] for edit in edit_steps],
        "alpha_levels": [edit["alpha_level"] for edit in edit_steps],
        "lpips_edited_recon": compute_lpips(lpips_fn, edited_img, manipulator, s_recon),
        "edit_steps": edit_steps,
    }


def compute_lpips(lpips_fn, edited_img, manipulator, s_recon):
    if lpips_fn is None:
        return None
    try:
        transform = IMG_TRANSFORM
        recon_img = generate_from_s(manipulator, s_recon)
        with torch.no_grad():
            device = next(lpips_fn.parameters()).device
            edited_tensor = transform(edited_img).unsqueeze(0).to(device)
            recon_tensor = transform(recon_img).unsqueeze(0).to(device)
            return round(float(lpips_fn(edited_tensor, recon_tensor).item()), 4)
    except Exception as exc:
        print(f"  [lpips] {exc}")
        return None


def worker(rank: int, world_size: int, plans: list, done_keys: set, results_lock, results_path: str, failed_path: str, celeba_img_dir: str, edited_dir: str, progress_queue, batch_size: int = 8):
    import lpips as lpips_lib

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    e4e = load_e4e(device=device)
    clip_model, manipulator, fs3 = load_styleclip_global(device=device)
    direction_cache = precompute_all_directions(clip_model, manipulator, fs3)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    saver = AsyncSaver(num_threads=4)

    img_to_plans = group_rank_plans(plans, done_keys, rank, world_size)
    results_f = open(results_path + f".rank{rank}", "a")
    failed_f = open(failed_path + f".rank{rank}", "a")

    for paths, plan_groups in iter_image_batches(img_to_plans, celeba_img_dir, progress_queue, batch_size):
        valid_paths, w_list = encode_batch(e4e, paths, device)
        for img_path, w_plus in zip(valid_paths, w_list):
            for plan in plan_groups[img_path]:
                write_plan_result(
                    plan,
                    img_path,
                    w_plus,
                    e4e,
                    manipulator,
                    direction_cache,
                    edited_dir,
                    saver,
                    lpips_fn,
                    results_f,
                    failed_f,
                    results_lock,
                    progress_queue,
                )

    saver.stop()
    results_f.close()
    failed_f.close()


def group_rank_plans(plans, done_keys, rank, world_size):
    img_to_plans = defaultdict(list)
    for plan in plans[rank::world_size]:
        key = f"{plan['image']}_v{plan.get('variant', 0)}"
        if key not in done_keys:
            img_to_plans[plan["image"]].append(plan)
    return img_to_plans


def iter_image_batches(img_to_plans, celeba_img_dir, progress_queue, batch_size):
    img_list = list(img_to_plans.keys())
    for start in range(0, len(img_list), batch_size):
        paths = []
        plan_groups = {}
        for img_name in img_list[start:start + batch_size]:
            path = os.path.join(celeba_img_dir, img_name)
            if not os.path.exists(path):
                for _ in img_to_plans[img_name]:
                    progress_queue.put(("skip", None))
                continue
            paths.append(path)
            plan_groups[path] = img_to_plans[img_name]
        if paths:
            yield paths, plan_groups


def write_plan_result(plan, img_path, w_plus, e4e, manipulator, direction_cache, edited_dir, saver, lpips_fn, results_f, failed_f, results_lock, progress_queue):
    variant = plan.get("variant", 0)
    img_name = plan["image"]
    try:
        rec = execute_plan(
            e4e,
            manipulator,
            direction_cache,
            img_path=img_path,
            w_plus=w_plus,
            edit_plan=plan["edits"],
            output_dir=edited_dir,
            variant=variant,
            saver=saver,
            lpips_fn=lpips_fn,
        )
        with results_lock:
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results_f.flush()
        progress_queue.put(("ok", None))
    except Exception as exc:
        with results_lock:
            failed_f.write(json.dumps({
                "image": img_name,
                "variant": variant,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False) + "\n")
            failed_f.flush()
        progress_queue.put(("fail", None))


def merge_rank_files(base: str, world_size: int) -> None:
    with open(base, "a") as out:
        for rank in range(world_size):
            part = base + f".rank{rank}"
            if not os.path.exists(part):
                continue
            with open(part) as f:
                for line in f:
                    if line.strip():
                        out.write(line.strip() + "\n")
            os.remove(part)


def launch_workers(args, dataset_plan, done_keys):
    world_size = min(args.num_gpus, torch.cuda.device_count())
    total_todo = sum(
        1 for plan in dataset_plan
        if f"{plan['image']}_v{plan.get('variant', 0)}" not in done_keys
    )
    print(f"GPU={world_size}  todo={total_todo}")

    os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
    results_lock = mp.Lock()
    progress_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        process = mp.Process(
            target=worker,
            args=(
                rank,
                world_size,
                dataset_plan,
                done_keys,
                results_lock,
                args.results_json,
                args.failed_json,
                args.celeba_img_dir,
                args.edited_dir,
                progress_queue,
                args.batch_size,
            ),
            daemon=True,
        )
        process.start()
        processes.append(process)

    ok, fail, skip = track_progress(processes, progress_queue, total_todo)
    for process in processes:
        process.join()

    merge_rank_files(args.results_json, world_size)
    merge_rank_files(args.failed_json, world_size)
    print(f"\nDone: ok={ok}  fail={fail}  skip={skip}")
    print(f"Results: {args.results_json}")


def track_progress(processes, progress_queue, total_todo):
    ok, fail, skip = 0, 0, 0
    pbar = tqdm(total=total_todo, desc="GAN generation", unit="sample")
    while any(process.is_alive() for process in processes):
        try:
            status, _ = progress_queue.get(timeout=1)
            if status == "ok":
                ok += 1
                pbar.update(1)
            elif status == "fail":
                fail += 1
            elif status == "skip":
                skip += 1
            pbar.set_postfix(ok=ok, fail=fail, skip=skip, refresh=False)
        except Exception:
            pass
    pbar.close()
    return ok, fail, skip
