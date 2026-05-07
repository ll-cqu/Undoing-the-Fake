# Weight adjustment:

# W_DETECTION 0.10→0.05，W_ATTR 0.80→1.80，W_ARC 0.05→0.10
# Hard-sample upsampling:

# GRPODataset adds two extra copies for samples with num_edits >= 4; total size grows by ~20-30%.
# Recall penalty for multi-edit samples (r_attr):

# When num_edits >= 4, low-recall predictions are downweighted: score x (0.5 + 0.5 x recall).
# recall=1.0 -> no penalty; recall=0.5 -> 75% score; recall=0.0 -> half score.
# difficulty weighting（compute_reward）：

# 3 edits -> 1.2x, 4 edits -> 1.4x, 5 edits -> 1.6x, giving hard samples larger gradients.

import os, sys, json, re, random, argparse, logging
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
_reward_logger: logging.Logger = None

def _get_reward_logger(output_dir: str) -> logging.Logger:
    global _reward_logger
    if _reward_logger is not None:
        return _reward_logger
    logger = logging.getLogger("reward")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(output_dir, "reward_log.jsonl"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    _reward_logger = logger
    return logger

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
WORKSPACE      = os.environ.get("UTF_WORKSPACE", ".")
_LOG_OUTPUT_DIR = ""
_VIS_ENABLED    = False
_VIS_DIR        = ""

def _init_vis(vis_dir: str):
    global _VIS_ENABLED, _VIS_DIR
    _VIS_ENABLED = True
    _VIS_DIR     = vis_dir
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[vis] enabled: {vis_dir}", flush=True)

CHECKPOINT     = os.environ.get("UTF_WARMUP_CHECKPOINT", "output_file/warmup_checkpoint")
STYLECLIP_ROOT = os.environ.get("UTF_STYLECLIP_ROOT", f"{WORKSPACE}/face_editing/StyleCLIP")
E4E_ROOT       = os.environ.get("UTF_E4E_ROOT", f"{WORKSPACE}/face_editing/encoder4editing")
STYLEGAN_PKL   = os.environ.get("UTF_STYLEGAN_PKL", f"{STYLECLIP_ROOT}/global_torch/model/ffhq.pkl")
FS3_PATH       = os.environ.get("UTF_FS3_PATH", f"{STYLECLIP_ROOT}/global_torch/npy/ffhq/fs3.npy")
E4E_CKPT       = os.environ.get("UTF_E4E_CKPT", f"{WORKSPACE}/face_editing/pretrained/e4e_ffhq_encode.pt")
BASE_MODEL     = os.environ.get("UTF_BASE_MODEL", "hf_models/Qwen3-VL-8B-Instruct")

# ─────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────
W_FORMAT    = 0.05
W_DETECTION = 0.05   # Multi-edit samples almost always get is_manipulated right, so lower this weight
W_ATTR      = 1.80   # Increase this to prioritize attr/pair prediction on multi-edit samples
W_LEVEL     = 0.20
W_ARC       = 0.10   # Increase this so ArcFace recovery contributes meaningfully to gradients

ALPHA_LEVEL_TO_MIDPOINT = {
    "moderate": 1.00,
    "strong":   1.40,
    "extreme":  1.80,
}

BETA_DEFAULT = 0.10
ATTRIBUTE_CONFIG = {
    "smile":      {"neutral": "face without smile",          "target": "smiling face"},
    "hair_curly": {"neutral": "face with straight hair",     "target": "face with curly hair",   "beta_override": 0.08},
    "hair_bangs": {"neutral": "face without bangs",          "target": "face with bangs",         "beta_override": 0.08},
    "beard":      {"neutral": "face without beard",          "target": "face with beard"},
    "makeup":     {"neutral": "face without makeup",         "target": "face with full makeup"},
    "age":        {"neutral": "young face",                  "target": "old face"},
    "skin_tan":   {"neutral": "face with fair skin",         "target": "face with tanned skin"},
    "chubby":     {"neutral": "face with normal face shape", "target": "chubby face"},
    "eye_big":    {"neutral": "face with normal sized eyes", "target": "face with big eyes",      "beta_override": 0.09},
}

# ─────────────────────────────────────────────
# GAN models (per-device cache)
# Use arcface_utils instead of arc_session.
# ─────────────────────────────────────────────
_GAN_CACHE: dict = {}

def _legacy_get_gan_models(device: str):
    if device in _GAN_CACHE:
        return _GAN_CACHE[device]

    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    import time

    _extra = [STYLECLIP_ROOT, E4E_ROOT, os.path.join(STYLECLIP_ROOT, "global_torch")]
    _orig_path = sys.path.copy()
    for p in _extra:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import clip
        from models.psp import pSp
        from argparse import Namespace as NS
        from manipulate import Manipulator, LoadModel
        from StyleCLIP import GetDt, GetBoundary

        t0 = time.time()

        ckpt = torch.load(E4E_CKPT, map_location="cpu")
        opts = NS(**{**ckpt["opts"], "checkpoint_path": E4E_CKPT, "device": device})
        e4e  = pSp(opts).eval().to(device)

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()

        M = Manipulator(dataset_name="ffhq")
        M.device = device
        M.G = LoadModel(STYLEGAN_PKL, device)
        M.SetGParameters()
        M.GenerateS(num_img=10000)
        M.GetCodeMS()

        fs3        = np.load(FS3_PATH)
        directions = {}
        for attr, cfg in ATTRIBUTE_CONFIG.items():
            beta = cfg.get("beta_override", BETA_DEFAULT)
            dt   = GetDt([cfg["target"], cfg["neutral"]], clip_model)
            boundary, _ = GetBoundary(fs3, dt, M, threshold=beta)
            directions[attr] = boundary
            print(f"[GAN {device}] direction: {attr}", flush=True)

        # Warm up ArcFace through arcface_utils.
        get_arcface(device)
        print(f"[GAN {device}] ArcFace ready via arcface_utils", flush=True)

        print(f"[GAN {device}] init done in {time.time()-t0:.1f}s", flush=True)
        _GAN_CACHE[device] = (e4e, M, directions)

    finally:
        sys.path = _orig_path

    return _GAN_CACHE[device]


# ─────────────────────────────────────────────
# GAN helpers
# ─────────────────────────────────────────────
_IMG_TF = T.Compose([
    T.Resize((256, 256)), T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def _encode(e4e, img: Image.Image, device: str):
    x = _IMG_TF(img.convert("RGB")).unsqueeze(0).to(device)
    _, w = e4e(x, randomize_noise=False, return_latents=True)
    return w

@torch.no_grad()
def _get_s(M, w) -> list:
    return M.S2List(M.G.synthesis.W2S(w))

def _apply_boundary(M, s, boundary, alpha: float) -> list:
    M.num_images = 1
    M.alpha      = [alpha]
    M.step       = 1
    codes = M.MSCode(s, boundary)
    return [c[:, 0, :] for c in codes]

@torch.no_grad()
def _generate(M, s) -> Image.Image:
    M.num_images = 1
    M.alpha      = [0]
    M.step       = 1
    codes = M.MSCode(s, [np.zeros_like(b) for b in s])
    return Image.fromarray(M.GenerateImg(codes)[0, 0])


def _legacy_run_recovery(
    edited_img:          Image.Image,
    recon_img:           Image.Image,
    pred_manipulations:  list,
    device:              str,
) -> tuple:
    """
    Return (arc_score, recovered_img), matching train_grpo_react.py.
    Similarity is computed with arcface_utils.arcface_sim.
    """
    try:
        e4e, M, directions = get_gan_models(device)
        get_arcface(device)

        s = _get_s(M, _encode(e4e, edited_img, device))
        for m in pred_manipulations:
            attr   = m.get("attr")
            action = m.get("action")
            level  = m.get("alpha_level")
            if attr not in directions or not action or not level:
                continue
            alpha_abs      = ALPHA_LEVEL_TO_MIDPOINT.get(level, 1.0)
            recovery_alpha = -alpha_abs if action == "add" else +alpha_abs
            s = _apply_boundary(M, s, directions[attr], recovery_alpha)

        recovered = _generate(M, s)
        arc_score = arcface_sim(recon_img, recovered, device)
        return arc_score, recovered

    except Exception as e:
        print(f"[recovery] {e}", flush=True)
        return None, edited_img


# ─────────────────────────────────────────────
# Model output parsing
# ─────────────────────────────────────────────
def _legacy_parse_output(text: str):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None, []
        data           = json.loads(m.group())
        is_manipulated = data.get("is_manipulated", None)
        manipulations  = []
        for item in data.get("manipulations", []):
            attr   = item.get("attr")
            action = item.get("action")
            level  = item.get("alpha_level", "")
            if attr and action:
                manipulations.append({
                    "attr":        attr,
                    "action":      action,
                    "alpha_level": level,
                })
        return is_manipulated, manipulations
    except Exception:
        return None, []


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            c = msg.get("content", "")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts.extend(x.get("text", "") for x in c if x.get("type") == "text")
        return " ".join(parts)
    return str(completion)


from undo_the_fake.models.gan import get_arcface as get_arcface, get_gan_models as get_gan_models, run_arc_recovery as run_recovery
from undo_the_fake.utils.parsing import extract_completion_text as _extract_text, parse_output as parse_output


# ─────────────────────────────────────────────
# Reward components
# ─────────────────────────────────────────────
def r_detection(pred_flag, gt_flag: bool) -> float:
    if pred_flag is None:
        return 0.0
    return 1.0 if bool(pred_flag) == bool(gt_flag) else 0.0


def r_format(pred_text: str) -> float:
    try:
        m = re.search(r'\{.*\}', pred_text, re.DOTALL)
        if not m:
            return 0.0
        data = json.loads(m.group())
        if "is_manipulated" not in data:
            return 0.5
        if data["is_manipulated"] and "manipulations" not in data:
            return 0.5
        return 1.0
    except Exception:
        return 0.0


_LEVEL_ORDER = {"moderate": 0, "strong": 1, "extreme": 2}


def r_attr(pred_manipulations: list, gt_attrs: list, gt_actions: list) -> float:
    gt_pairs     = set(zip(gt_attrs, gt_actions))
    gt_attrs_set = set(gt_attrs)
    pred_pairs   = {(m["attr"], m["action"])
                    for m in pred_manipulations if m.get("attr") and m.get("action")}
    pred_attrs   = {m["attr"] for m in pred_manipulations if m.get("attr")}

    def fbeta(tp, n_pred, n_gt, beta):
        if n_pred == 0 or n_gt == 0:
            return 0.0 if (n_pred + n_gt) > 0 else 1.0
        prec = tp / n_pred
        rec  = tp / n_gt
        if prec + rec == 0:
            return 0.0
        return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)

    pair_f2 = fbeta(len(gt_pairs & pred_pairs), len(pred_pairs), len(gt_pairs), beta=2.0)
    attr_f1 = fbeta(len(gt_attrs_set & pred_attrs), len(pred_attrs), len(gt_attrs_set), beta=1.0)
    base_score = 0.8 * pair_f2 + 0.2 * attr_f1

    # Penalize missed predictions on multi-edit samples (>=4): lower recall reduces the score.
    n_gt = len(gt_attrs)
    if n_gt >= 4:
        recall = len(gt_pairs & pred_pairs) / len(gt_pairs) if gt_pairs else 1.0
        base_score = base_score * (0.5 + 0.5 * recall)

    return base_score


def r_level(pred_manipulations: list, gt_attrs: list, gt_actions: list, gt_levels: list) -> float:
    gt_map   = {(a, act): lv for a, act, lv in zip(gt_attrs, gt_actions, gt_levels)}
    pred_map = {(m["attr"], m["action"]): m.get("alpha_level", "")
                for m in pred_manipulations if m.get("attr") and m.get("action")}
    hit_pairs = set(gt_map.keys()) & set(pred_map.keys())
    if not hit_pairs:
        return 0.0
    scores = []
    for pair in hit_pairs:
        gt_lv, pred_lv = gt_map[pair], pred_map[pair]
        if pred_lv == gt_lv:
            scores.append(1.0)
        elif pred_lv in _LEVEL_ORDER and gt_lv in _LEVEL_ORDER:
            scores.append(max(0.0, 1.0 - 0.5 * abs(_LEVEL_ORDER[pred_lv] - _LEVEL_ORDER[gt_lv])))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def r_arc_recovery(arc_recovered: Optional[float], arc_edited: Optional[float]) -> float:
    if arc_recovered is None or arc_edited is None:
        return 0.0
    denom = 1.0 - arc_edited + 1e-6
    rate  = (arc_recovered - arc_edited) / denom
    return float(np.clip(rate, 0.0, 1.0))


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def save_vis(edited_img, recovered_img, recon_img, img_path, step_ts, recovery_rate, pred_pairs, gt_pairs):
    try:
        W, H = 256, 256
        canvas = Image.new("RGB", (W * 3 + 20, H + 40), (30, 30, 30))
        for i, img in enumerate([recon_img, edited_img, recovered_img]):
            canvas.paste(img.resize((W, H)), (i * (W + 10), 20))
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        ts_clean = step_ts.replace(" ", "_").replace(":", "-")
        rate_str = f"{recovery_rate:.3f}".replace(".", "p")
        canvas.save(os.path.join(_VIS_DIR, f"{ts_clean}_{stem}_arc{rate_str}.png"))
    except Exception as e:
        print(f"[vis] save failed: {e}", flush=True)


# ─────────────────────────────────────────────
# Combined reward
# ─────────────────────────────────────────────
def compute_reward(
    completions:  list,
    image_path:   list,
    gt_attrs:     list,
    gt_actions:   list,
    gt_levels:    list,
    recon_path:   list,
    arc_edited:   list,
    **kwargs,
) -> list:
    if torch.distributed.is_initialized():
        rank   = torch.distributed.get_rank()
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        rank   = 0
        device = "cuda:0"

    rank0   = (not torch.distributed.is_initialized() or rank == 0)
    rewards = []
    import time
    step_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for pred_text, img_path, g_attrs, g_actions, g_levels, r_path, arc_edit_val in zip(
        completions, image_path, gt_attrs, gt_actions, gt_levels, recon_path, arc_edited
    ):
        pred_text = _extract_text(pred_text)
        pred_flag, pred_manipulations = parse_output(pred_text)

        gt_flag = True
        rf  = r_format(pred_text)
        rd  = r_detection(pred_flag, gt_flag)
        ra  = r_attr(pred_manipulations, g_attrs, g_actions)
        rl  = r_level(pred_manipulations, g_attrs, g_actions, g_levels)

        rarc    = 0.0
        arc_rec = None

        run_recovery_flag = (
            pred_flag is True
            and pred_manipulations
            and img_path and os.path.exists(img_path)
            and r_path   and os.path.exists(r_path)
            and arc_edit_val is not None
        )
        if run_recovery_flag:
            try:
                Image.open(img_path).verify()
                Image.open(r_path).verify()
            except Exception as ve:
                print(f"[reward/arc] corrupted image, skip: {ve}", flush=True)
                run_recovery_flag = False

        if run_recovery_flag:
            try:
                edited_img = Image.open(img_path).convert("RGB")
                recon_img  = Image.open(r_path).convert("RGB")
                arc_rec, recovered_img = run_recovery(edited_img, recon_img, pred_manipulations, device)
                rarc = r_arc_recovery(arc_rec, arc_edit_val)

                if _VIS_ENABLED and rank0 and recovered_img is not None:
                    save_vis(
                        edited_img, recovered_img, recon_img,
                        img_path, step_ts, rarc,
                        sorted({(m["attr"], m["action"]) for m in pred_manipulations}),
                        sorted(set(zip(g_attrs, g_actions))),
                    )
            except Exception as e:
                print(f"[reward/arc] {e}", flush=True)

        total = (
            W_FORMAT    * rf
            + W_DETECTION * rd
            + W_ATTR      * ra
            + W_LEVEL     * rl
            + W_ARC       * rarc
        )
        # Give hard multi-edit samples higher weight and larger gradients.
        n_edits = len(g_attrs)
        difficulty_weight = 1.0 + 0.2 * max(0, n_edits - 2)  # 3 edits -> 1.2x, 5 edits -> 1.6x
        total = total * difficulty_weight
        rewards.append(float(total))

        if rank0 and _LOG_OUTPUT_DIR:
            gt_pairs   = set(zip(g_attrs, g_actions))
            pred_pairs = {(m["attr"], m["action"])
                          for m in pred_manipulations if m.get("attr") and m.get("action")}
            tp      = len(gt_pairs & pred_pairs)
            fp      = len(pred_pairs - gt_pairs)
            fn      = len(gt_pairs - pred_pairs)
            tp_attr = len(set(g_attrs) & {m["attr"] for m in pred_manipulations})

            logger = _get_reward_logger(_LOG_OUTPUT_DIR)
            log_entry = {
                "ts":    step_ts, "rank": rank,
                "image": os.path.basename(img_path),
                "pred_flag": pred_flag,
                "pred_pairs": sorted(pred_pairs),
                "pred_levels": {m["attr"]: m.get("alpha_level") for m in pred_manipulations},
                "gt_pairs": sorted(gt_pairs),
                "gt_levels": dict(zip(g_attrs, g_levels)),
                "r_fmt": round(rf, 4), "r_det": round(rd, 4),
                "r_attr": round(ra, 4), "r_level": round(rl, 4),
                "r_arc_recovery": round(rarc, 4),
                "difficulty_weight": round(difficulty_weight, 2),
                "total": round(total, 4),
                "arc_recon_edited": arc_edit_val,
                "arc_recon_recovered": round(arc_rec, 4) if arc_rec is not None else None,
                "tp": tp, "fp": fp, "fn": fn, "tp_attr": tp_attr,
                "pred_text": pred_text[:300],
            }
            logger.debug(json.dumps(log_entry, ensure_ascii=False))
            logger.info(
                f"  [R] fmt={rf:.2f} det={rd:.2f} "
                f"attr={ra:.2f}(tp={tp},fp={fp},fn={fn}|attr_tp={tp_attr}) "
                f"lv={rl:.2f} arc={rarc:.3f} -> {total:.3f} | "
                f"pred={sorted(pred_pairs)} gt={sorted(gt_pairs)}"
            )

    return rewards


# ─────────────────────────────────────────────
# System / user prompt (consistent with warmup training)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a face manipulation detection expert. "
    "Given an image, determine whether it has been manipulated. "
    "If manipulated, identify each altered facial attribute, "
    "the direction of change (add/remove), and the manipulation strength "
    "(moderate/strong/extreme). "
    "Respond ONLY with a valid JSON object."
)

USER_PROMPTS = [
    (
        "Analyze this face image and determine if it has been manipulated. "
        "If so, identify each altered attribute, the direction (add/remove), "
        "and strength (moderate/strong/extreme).\n"
        'Reply with JSON only: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    (
        "Is this face image authentic or has it been edited? "
        "List any manipulated facial attributes with their edit direction and intensity.\n"
        'Output JSON: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    (
        "Detect facial attribute manipulations in this image. "
        "Return a JSON object with is_manipulated (bool) and a list of manipulations, "
        "each with attr, action (add/remove), and alpha_level (moderate/strong/extreme)."
    ),
    (
        "This face image may have been tampered with. "
        "Identify all edited facial attributes so that the edits can be reversed. "
        "For each, provide the attribute name, the edit action (add/remove), "
        "and the manipulation strength (moderate/strong/extreme).\n"
        'JSON output only: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
    (
        "Perform a deepfake detection analysis on this face image. "
        "Report whether manipulation is detected and describe each altered attribute "
        "with its action (add/remove) and alpha level (moderate/strong/extreme).\n"
        'Output format — JSON only: {"is_manipulated": bool, "manipulations": '
        '[{"attr": str, "action": "add"|"remove", "alpha_level": "moderate"|"strong"|"extreme"}]}'
    ),
]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class GRPODataset(Dataset):
    def __init__(self, jsonl_path: str):
        base_records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    base_records.append(json.loads(line))

        # Upsample hard samples with num_edits >= 4 by 3x.
        hard_records = [r for r in base_records
                        if r.get("_meta", {}).get("num_edits", 0) >= 4]
        self.records = base_records + hard_records * 2
        random.shuffle(self.records)
        print(f"[dataset] base={len(base_records)}  hard={len(hard_records)}  "
              f"total={len(self.records)}", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        meta = rec.get("_meta", {})

        edited_path = meta.get("edited_path", "")
        try:
            img = Image.open(edited_path).convert("RGB")
            img.thumbnail((336, 336))
        except Exception:
            img = Image.new("RGB", (336, 336), (128, 128, 128))

        user_prompt = random.choice(USER_PROMPTS)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        return {
            "prompt":      messages,
            "image":       img,
            "image_path":  edited_path,
            "gt_attrs":    meta.get("attrs",        []),
            "gt_actions":  meta.get("actions",      []),
            "gt_levels":   meta.get("alpha_levels", []),
            "recon_path":  meta.get("recon_path",   ""),
            "arc_edited":  meta.get("arc_recon_edited"),
        }


# ─────────────────────────────────────────────
# Qwen3-VL patch
# ─────────────────────────────────────────────
def _find_rope_index_owner(model):
    visited = set()
    def _search(obj):
        if id(obj) in visited:
            return None
        visited.add(id(obj))
        if "get_rope_index" in type(obj).__dict__:
            return obj
        for attr in ("base_model", "model"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                result = _search(child)
                if result is not None:
                    return result
        return None
    return _search(model)


def _legacy_patch_qwen3vl_forward(model):
    _orig_forward = model.forward
    def _patched_forward(**kwargs):
        mtt = kwargs.get("mm_token_type_ids")
        am  = kwargs.get("attention_mask")
        if mtt is not None and am is not None:
            seq_len = am.shape[1]
            cur_len = mtt.shape[1]
            if cur_len < seq_len:
                pad = torch.zeros(mtt.shape[0], seq_len - cur_len,
                                  dtype=mtt.dtype, device=mtt.device)
                kwargs["mm_token_type_ids"] = torch.cat([pad, mtt], dim=1)
            elif cur_len > seq_len:
                kwargs["mm_token_type_ids"] = mtt[:, cur_len - seq_len:]
        return _orig_forward(**kwargs)
    model.forward = _patched_forward

    owner = _find_rope_index_owner(model)
    if owner is None:
        print("[patch] WARNING: could not find get_rope_index owner.", flush=True)
        return
    print(f"[patch] Patching get_rope_index on {type(owner).__name__}", flush=True)
    _orig_get_rope_index = type(owner).get_rope_index

    def _patched_get_rope_index(
        self_inner, input_ids,
        image_grid_thw=None, video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None, position_ids=None,
        mm_token_type_ids=None, **kwargs,
    ):
        kwargs.pop("position_ids", None)
        if attention_mask is None or attention_mask.all():
            return _orig_get_rope_index(
                self_inner, input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
        bsz, seq_len = input_ids.shape
        rp_ids = input_ids.new_zeros(bsz, seq_len)
        rp_am  = attention_mask.new_zeros(bsz, seq_len)
        rp_mtt = (mm_token_type_ids.new_zeros(bsz, seq_len)
                  if mm_token_type_ids is not None else None)
        real_lens = []
        for b in range(bsz):
            mask_b   = attention_mask[b].bool()
            real_len = int(mask_b.sum().item())
            real_lens.append(real_len)
            rp_ids[b, :real_len] = input_ids[b, mask_b]
            rp_am [b, :real_len] = 1
            if rp_mtt is not None:
                rp_mtt[b, :real_len] = mm_token_type_ids[b, mask_b]

        try:
            pos_ids, rope_deltas = _orig_get_rope_index(
                self_inner, rp_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=rp_am,
                mm_token_type_ids=rp_mtt,
                **kwargs,
            )
        except RuntimeError as e:
            print(f"[patch] get_rope_index fallback: {e}", flush=True)
            pos_ids = torch.arange(seq_len, device=input_ids.device) \
                        .unsqueeze(0).unsqueeze(0).expand(3, bsz, seq_len).clone()
            rope_deltas = torch.zeros(bsz, 1, dtype=torch.long, device=input_ids.device)
            return pos_ids, rope_deltas

        lp_pos_ids = pos_ids.new_zeros(3, bsz, seq_len)
        for b, real_len in enumerate(real_lens):
            if real_len > 0:
                lp_pos_ids[:, b, seq_len - real_len:] = pos_ids[:, b, :real_len]
        return lp_pos_ids, rope_deltas

    type(owner).get_rope_index = _patched_get_rope_index


from undo_the_fake.models.qwen import patch_qwen3vl_forward as _patch_qwen3vl_forward


# ─────────────────────────────────────────────
# Eval callback
# ─────────────────────────────────────────────
class GRPOEvalCallback:
    def __init__(self, model, processor, eval_records, output_dir, eval_steps, device):
        self.model        = model
        self.processor    = processor
        self.eval_records = eval_records
        self.output_dir   = output_dir
        self.eval_steps   = eval_steps
        self.device       = device

    def on_step_end(self, step: int):
        if step == 0 or step % self.eval_steps != 0:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        import time
        print(f"\n[EvalCallback] step={step} evaluating {len(self.eval_records)} samples...",
              flush=True)

        self.model.eval()
        det_scores, hier_rows = [], []
        _model = self.model.module if hasattr(self.model, "module") else self.model

        for rec in self.eval_records:
            messages    = rec["messages"]
            meta        = rec.get("_meta", {})
            sample_type = meta.get("sample_type", "fake")
            gt_attrs    = meta.get("attrs",        [])
            gt_actions  = meta.get("actions",      [])
            gt_levels   = meta.get("alpha_levels", [])

            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if asst_msg is None:
                gt_flag = (sample_type == "fake")
            else:
                asst_content = asst_msg["content"]
                if isinstance(asst_content, list):
                    asst_content = " ".join(
                        c.get("text", "") for c in asst_content if c.get("type") == "text")
                try:
                    gt_flag = json.loads(asst_content).get("is_manipulated", sample_type == "fake")
                except Exception:
                    gt_flag = (sample_type == "fake")

            image_path = ""
            for msg in messages:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for c in content:
                    if c.get("type") == "image" and isinstance(c.get("image"), str):
                        image_path = c["image"]
                        break
                if image_path:
                    break

            try:
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((512, 512))

                user_msgs = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",   "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text":  USER_PROMPTS[0]},
                    ]},
                ]
                inputs = self.processor.apply_chat_template(
                    user_msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    out_ids = _model.generate(**inputs, max_new_tokens=512, do_sample=False)
                generated = self.processor.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                pred_flag, pred_list = parse_output(generated)
                det_scores.append(1.0 if pred_flag == gt_flag else 0.0)

                if gt_flag and gt_attrs:
                    gt_attrs_set = set(gt_attrs)
                    pred_attrs   = {m["attr"] for m in pred_list if m.get("attr")}
                    tp_attrs     = gt_attrs_set & pred_attrs
                    attr_f1 = (2 * len(tp_attrs) / (len(pred_attrs) + len(gt_attrs_set))
                               if (pred_attrs or gt_attrs_set) else 0.0)
                    gt_act   = dict(zip(gt_attrs, gt_actions))
                    pred_act = {m["attr"]: m["action"] for m in pred_list}
                    action_ok  = {a for a in tp_attrs if pred_act.get(a) == gt_act[a]}
                    action_acc = len(action_ok) / len(tp_attrs) if tp_attrs else 0.0
                    gt_lv    = dict(zip(gt_attrs, gt_levels))
                    pred_lv  = {m["attr"]: m.get("alpha_level") for m in pred_list}
                    level_acc = (sum(pred_lv.get(a) == gt_lv[a] for a in action_ok)
                                 / len(action_ok) if action_ok else 0.0)
                    hier_rows.append({"attr_f1": attr_f1, "action_acc": action_acc, "level_acc": level_acc})

            except Exception as e:
                print(f"[EvalCallback] SKIP {image_path!r}: {e}", flush=True)

        torch.cuda.empty_cache()
        self.model.train()

        def mean(lst): return round(float(np.mean(lst)), 4) if lst else 0.0

        metrics = {
            "step":          step,
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_evaluated":   len(det_scores),
            "detection_acc": mean(det_scores),
            "attr_f1":       mean([h["attr_f1"]    for h in hier_rows]),
            "action_acc":    mean([h["action_acc"] for h in hier_rows]),
            "level_acc":     mean([h["level_acc"]  for h in hier_rows]),
        }

        with open(os.path.join(self.output_dir, "eval_results.jsonl"), "a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print(
            f"[EvalCallback step={step}] "
            f"evaluated={metrics['n_evaluated']}  "
            f"det={metrics['detection_acc']:.4f}  "
            f"attr_f1={metrics['attr_f1']:.4f}  "
            f"action={metrics['action_acc']:.4f}  "
            f"level={metrics['level_acc']:.4f}",
            flush=True,
        )


# ─────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl",     default="data/grpo/grpo_gan/grpo_train.jsonl")
    parser.add_argument("--checkpoint",      default=CHECKPOINT)
    parser.add_argument("--output_dir",      default="output_file/train")
    parser.add_argument("--max_steps",       type=int,   default=8000)
    parser.add_argument("--batch_size",      type=int,   default=1)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=5e-7)
    parser.add_argument("--num_generations", type=int,   default=4)
    parser.add_argument("--max_new_tokens",  type=int,   default=512)
    parser.add_argument("--lora_r",          type=int,   default=64)
    parser.add_argument("--lora_alpha",      type=int,   default=128)
    parser.add_argument("--save_steps",      type=int,   default=100)
    parser.add_argument("--logging_steps",   type=int,   default=10)
    parser.add_argument("--eval_jsonl",      default="data/warmup/warmup_test.jsonl")
    parser.add_argument("--eval_steps",      type=int,   default=100)
    parser.add_argument("--eval_samples",    type=int,   default=200)
    parser.add_argument("--vis",             action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    from undo_the_fake.training.grpo_trainer import GRPOTrainingJob

    GRPOTrainingJob(args=args, cfg=__import__(__name__, fromlist=[""])).run()


if __name__ == "__main__":
    main()
