"""
warmup.py - Qwen3-VL-8B LoRA SFT warmup on one 8-GPU node.
Usage: accelerate launch warmup.py
"""

import json
import os
import re
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from PIL import Image

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_ID      = "hf_models/Qwen3-VL-8B-Instruct"
TRAIN_FILE    = "data/warmup/warmup_train.jsonl"
EVAL_FILE     = "data/warmup/warmup_test.jsonl"
OUTPUT_DIR    = "output_file/warmup"

LORA_R        = 64
LORA_ALPHA    = 128
LORA_DROPOUT  = 0.05

NUM_EPOCHS    = 5
PER_DEVICE_BS = 2
GRAD_ACCUM    = 4
LR            = 9e-5
WARMUP_RATIO  = 0.03
MAX_SEQ_LEN   = 2048
SAVE_STEPS    = 300
EVAL_STEPS    = 300
LOG_STEPS     = 10

ALPHA_LEVELS  = ["moderate", "strong", "extreme"]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = (
    "You are a face manipulation detection expert. "
    "Given an image, determine whether it has been manipulated. "
    "If manipulated, identify each altered facial attribute, "
    "the direction of change (add/remove), and the manipulation strength "
    "(moderate/strong/extreme). "
    "Respond ONLY with a valid JSON object."
)


# ─────────────────────────────────────────────
# Model output parsing
# ─────────────────────────────────────────────
def _legacy_parse_output(text: str):
    """
    Parse JSON from model output.

    Returns:
        is_manipulated : bool | None
        manipulations  : [{"attr": str, "action": str, "alpha_level": str}, ...]
    """
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None, []
        obj            = json.loads(match.group())
        is_manipulated = obj.get("is_manipulated", None)
        manipulations  = []
        for m in obj.get("manipulations", []):
            attr   = m.get("attr")
            action = m.get("action")
            level  = m.get("alpha_level", "")
            if attr and action:
                manipulations.append({
                    "attr":        attr,
                    "action":      action,
                    "alpha_level": level,
                })
        return is_manipulated, manipulations
    except Exception:
        return None, []


# ─────────────────────────────────────────────
# Metric functions (three-stage cascade)
# ─────────────────────────────────────────────
def _legacy_detection_acc(pred_flag, gt_flag) -> float:
    """Stage 0: per-sample binary accuracy for is_manipulated."""
    if pred_flag is None or gt_flag is None:
        return 0.0
    return 1.0 if pred_flag == gt_flag else 0.0


def _legacy_compute_hierarchical_metrics(pred_list: list, gt_list: list) -> dict:
    """
    Three-stage cascade metrics. Each stage depends on the previous stage.

    Stage 1 - attr F1: which attributes were predicted, ignoring action.
    Stage 2 - action_acc: add/remove correctness conditioned on stage-1 TP.
    Stage 3 - level_acc: strength correctness conditioned on stage-2 correctness.
    """
    gt_attrs   = {m["attr"] for m in gt_list}
    pred_attrs = {m["attr"] for m in pred_list}

    # Stage 1: attr F1
    tp_attrs       = gt_attrs & pred_attrs
    attr_precision = len(tp_attrs) / len(pred_attrs) if pred_attrs else 0.0
    attr_recall    = len(tp_attrs) / len(gt_attrs)   if gt_attrs   else 1.0
    denom_f1       = attr_precision + attr_recall
    attr_f1        = (2 * attr_precision * attr_recall / denom_f1
                      if denom_f1 > 0 else 0.0)

    # Stage 2: action_acc conditioned on stage-1 TP
    gt_action   = {m["attr"]: m["action"]      for m in gt_list}
    pred_action = {m["attr"]: m["action"]      for m in pred_list}
    if tp_attrs:
        action_correct_attrs = {a for a in tp_attrs
                                if pred_action.get(a) == gt_action[a]}
        action_acc           = len(action_correct_attrs) / len(tp_attrs)
    else:
        action_correct_attrs = set()
        action_acc           = 0.0

    # Stage 3: level_acc conditioned on stage-2 correctness
    gt_level   = {m["attr"]: m["alpha_level"] for m in gt_list}
    pred_level = {m["attr"]: m["alpha_level"] for m in pred_list}
    if action_correct_attrs:
        n_correct = sum(pred_level.get(a) == gt_level[a]
                        for a in action_correct_attrs)
        level_acc = n_correct / len(action_correct_attrs)
    else:
        level_acc = 0.0

    return {
        "attr_precision": round(attr_precision, 4),
        "attr_recall":    round(attr_recall,    4),
        "attr_f1":        round(attr_f1,        4),
        "action_acc":     round(action_acc,     4),
        "level_acc":      round(level_acc,      4),
    }


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def _legacy_normalize_messages(messages):
    for msg in messages:
        if isinstance(msg["content"], str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
    return messages


def _legacy_extract_assistant_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c["text"] for c in content if c.get("type") == "text")
    return ""


from undo_the_fake.configs.runtime import LORA_TARGET_MODULES as LORA_TARGET_MODULES, SYSTEM_PROMPT as SYSTEM_PROMPT
from undo_the_fake.utils.metrics import detection_acc as detection_acc, hierarchical_metrics as compute_hierarchical_metrics
from undo_the_fake.utils.parsing import (
    extract_text as extract_assistant_text,
    normalize_messages as normalize_messages,
    parse_output as parse_output,
)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        with open(jsonl_path) as f:
            self.records = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        messages = self.records[idx]["messages"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        messages = normalize_messages(messages)

        for msg in messages:
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c.get("type") == "image" and isinstance(c["image"], str):
                        img = Image.open(c["image"]).convert("RGB")
                        img.thumbnail((512, 512))
                        c["image"] = img

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

        squeezed = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            squeezed[k] = v.squeeze(0) if k != "image_grid_thw" else v.reshape(-1, 3)
        inputs = squeezed

        input_ids = inputs["input_ids"]
        labels    = input_ids.clone()
        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False)
        a_len     = len(assistant_tokens)
        mask_till = 0
        for i in range(len(input_ids) - a_len, -1, -1):
            if input_ids[i: i + a_len].tolist() == assistant_tokens:
                mask_till = i + a_len
                break
        labels[:mask_till] = -100
        inputs["labels"] = labels
        return inputs


# ─────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────
class DataCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids_out, attn_mask_out, labels_out, mm_type_ids_out = [], [], [], []
        pixel_values, image_grid_thw = [], []

        for b in batch:
            pad = max_len - b["input_ids"].shape[0]
            input_ids_out.append(torch.cat([
                b["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            attn_mask_out.append(torch.cat([
                b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels_out.append(torch.cat([
                b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            if "mm_token_type_ids" in b:
                mm_type_ids_out.append(torch.cat([
                    b["mm_token_type_ids"], torch.zeros(pad, dtype=torch.long)]))
            if "pixel_values" in b:
                pixel_values.append(b["pixel_values"])
            if "image_grid_thw" in b:
                thw = b["image_grid_thw"]
                image_grid_thw.append(thw.unsqueeze(0) if thw.dim() == 1 else thw)

        out = {
            "input_ids":      torch.stack(input_ids_out),
            "attention_mask": torch.stack(attn_mask_out),
            "labels":         torch.stack(labels_out),
        }
        if mm_type_ids_out:
            out["mm_token_type_ids"] = torch.stack(mm_type_ids_out)
        if pixel_values:
            out["pixel_values"]   = torch.cat(pixel_values,   dim=0)
            out["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        return out


# ─────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────
def make_compute_metrics(eval_records, processor, model_ref):
    import torch.distributed as dist

    _state_holder = [None]   # The StateCapture callback injects the current step.

    def compute_metrics(eval_pred):
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}

        device = next(model_ref.parameters()).device
        model_ref.eval()

        detection_scores = []   # Stage 0: is_manipulated
        hier_scores      = []   # Stages 1-3: hierarchical metrics for fake samples only
        samples          = []

        pbar = tqdm(eval_records, desc="Evaluating", ncols=100, leave=False)
        for rec in pbar:
            messages = rec["messages"]

            gt_text          = extract_assistant_text(messages[-1]["content"])
            gt_flag, gt_list = parse_output(gt_text)

            # Extract image path before image loading while it is still a string.
            image_path = ""
            for c in (messages[0]["content"] if isinstance(messages[0]["content"], list) else []):
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    image_path = c["image"]
                    break

            user_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[:-1]
            user_messages = normalize_messages(user_messages)

            # Load images.
            for msg in user_messages:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c.get("type") == "image" and isinstance(c["image"], str):
                            img = Image.open(c["image"]).convert("RGB")
                            img.thumbnail((512, 512))
                            c["image"] = img

            inputs = processor.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model_ref.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            generated = processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            pred_flag, pred_list = parse_output(generated)

            # Stage 0: detection
            detection_scores.append(detection_acc(pred_flag, gt_flag))

            # Stages 1-3: fake samples only
            if gt_flag:
                hier_scores.append(compute_hierarchical_metrics(pred_list, gt_list))

            # Sample record
            samples.append({
                "image":       image_path,
                "gt_text":     gt_text,
                "pred_text":   generated,
                "gt_flag":     gt_flag,
                "pred_flag":   pred_flag,
                "det_correct": gt_flag == pred_flag,
            })

            if detection_scores:
                pbar.set_postfix({"det_acc": f"{np.mean(detection_scores):.3f}"})

        torch.cuda.empty_cache()

        # Aggregate metrics.
        det_acc        = float(np.mean(detection_scores))                          if detection_scores else 0.0
        attr_precision = float(np.mean([h["attr_precision"] for h in hier_scores])) if hier_scores else 0.0
        attr_recall    = float(np.mean([h["attr_recall"]    for h in hier_scores])) if hier_scores else 0.0
        attr_f1        = float(np.mean([h["attr_f1"]        for h in hier_scores])) if hier_scores else 0.0
        action_acc     = float(np.mean([h["action_acc"]     for h in hier_scores])) if hier_scores else 0.0
        level_acc      = float(np.mean([h["level_acc"]      for h in hier_scores])) if hier_scores else 0.0

        step = _state_holder[0].global_step if _state_holder[0] else -1

        metrics = {
            "detection_acc":  round(det_acc,        4),
            "attr_precision": round(attr_precision,  4),
            "attr_recall":    round(attr_recall,     4),
            "attr_f1":        round(attr_f1,         4),
            "action_acc":     round(action_acc,      4),   # Conditioned on correct attr
            "level_acc":      round(level_acc,       4),   # Conditioned on correct action
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(os.path.join(OUTPUT_DIR, "eval_results.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "timestamp": ts, **metrics},
                               ensure_ascii=False) + "\n")

        sample_file = os.path.join(
            OUTPUT_DIR,
            f"eval_samples_step{step}_{ts.replace(' ', '_').replace(':', '-')}.jsonl"
        )
        with open(sample_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        tqdm.write(
            f"\n[Eval step={step}] "
            f"det={det_acc:.4f}  "
            f"attr_f1={attr_f1:.4f} (p={attr_precision:.4f} r={attr_recall:.4f})  "
            f"action={action_acc:.4f}  "
            f"level={level_acc:.4f}"
        )
        return metrics

    class StateCapture(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            _state_holder[0] = state

    return compute_metrics, StateCapture()


# ─────────────────────────────────────────────
# Training progress callback
# ─────────────────────────────────────────────
class TrainProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            tqdm.write(
                f"[Train] step={state.global_step}/{state.max_steps} | "
                f"epoch={logs.get('epoch', 0):.2f} | "
                f"loss={logs.get('loss', 0):.4f} | "
                f"lr={logs.get('learning_rate', 0):.2e}"
            )


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────
def main():
    from undo_the_fake.training.warmup_trainer import WarmupTrainer

    WarmupTrainer(cfg=__import__(__name__, fromlist=[""])).run()


if __name__ == "__main__":
    main()
