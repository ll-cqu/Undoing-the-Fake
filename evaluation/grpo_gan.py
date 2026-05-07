"""
eval_single_ckpt.py
Evaluate one checkpoint and write per-sample outputs to jsonl.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 \
      eval_grpo_gan.py \
      --ckpt output_file/train_checkpoint \
      --eval_jsonl data/warmup/warmup_test.jsonl \
      --output output_file/eval_samples.jsonl \
      --eval_samples 200
"""

import os, json, random, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from undo_the_fake.configs.runtime import FACE_DETECT_BASE_MODEL as BASE_MODEL, SYSTEM_PROMPT, USER_PROMPT
from undo_the_fake.utils.metrics import hier_metrics_from_lists as hier_metrics
from undo_the_fake.utils.parsing import parse_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         default="output_file/checkpoint")
    parser.add_argument("--eval_jsonl",   default="data/warmup/warmup_test.jsonl")
    parser.add_argument("--output",       default="output_file/eval_samples.jsonl")
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    from accelerate import Accelerator
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    accelerator = Accelerator()
    rank    = accelerator.process_index
    world   = accelerator.num_processes
    is_main = accelerator.is_main_process
    device  = str(accelerator.device)

    # ── load eval records ────────────────────────
    random.seed(args.seed)
    with open(args.eval_jsonl) as f:
        all_records = [json.loads(l) for l in f if l.strip()]
    if args.eval_samples and args.eval_samples > 0:
        random.shuffle(all_records)
        eval_records = all_records[:args.eval_samples]
    else:
        eval_records = all_records
    if is_main:
        print(f"Eval samples: {len(eval_records)}", flush=True)

    # ── load model ───────────────────────────────
    if is_main:
        print(f"Loading checkpoint: {args.ckpt}", flush=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=None, trust_remote_code=True,
    )
    model.config.use_cache = False

    if os.path.exists(os.path.join(args.ckpt, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, args.ckpt, is_trainable=False)
        model = model.merge_and_unload()
        if is_main:
            print("LoRA merged", flush=True)
    else:
        if is_main:
            print("No adapter_config.json, using base model as-is", flush=True)

    model.eval().to(device)

    # ── shard ────────────────────────────────────
    shard = eval_records[rank::world]
    local_rows = []

    for rec in tqdm(shard, desc=f"[rank{rank}]", ncols=100, disable=(rank != 0)):
        messages    = rec["messages"]
        meta        = rec.get("_meta", {})
        sample_type = meta.get("sample_type", "fake")
        gt_flag     = (sample_type == "fake")
        gt_attrs    = meta.get("attrs",        [])
        gt_actions  = meta.get("actions",      [])
        gt_levels   = meta.get("alpha_levels", [])
        num_edits   = meta.get("num_edits", len(gt_attrs))

        image_path = ""
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list): continue
            for c in content:
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    image_path = c["image"]; break
            if image_path: break

        row = {
            "image":       image_path,
            "sample_type": sample_type,
            "gt_flag":     gt_flag,
            "pred_flag":   None,
            "det_correct": None,
            "gt_attrs":    gt_attrs,
            "gt_actions":  gt_actions,
            "gt_levels":   gt_levels,
            "num_edits":   int(num_edits) if num_edits else len(gt_attrs),
            "pred_text":   "",
            "pred_manips": [],
            "attr_f1":     None,
            "action_acc":  None,
            "level_acc":   None,
        }

        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((512, 512))
            user_msgs = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text":  USER_PROMPT},
                ]},
            ]
            inputs = processor.apply_chat_template(
                user_msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

            pred_flag, pred_list = parse_output(text)
            row["pred_text"]   = text
            row["pred_flag"]   = pred_flag
            row["pred_manips"] = pred_list
            row["det_correct"] = (pred_flag == gt_flag)

            if gt_flag and gt_attrs:
                f1, act_acc, lv_acc = hier_metrics(pred_list, gt_attrs, gt_actions, gt_levels)
                row["attr_f1"]    = round(f1,      4)
                row["action_acc"] = round(act_acc, 4)
                row["level_acc"]  = round(lv_acc,  4)

        except Exception as e:
            if rank == 0:
                print(f"  SKIP {image_path}: {e}", flush=True)

        local_rows.append(row)

    # ── save per-rank tmp ────────────────────────
    tmp_path = f"/tmp/eval_single_rank{rank}.jsonl"
    with open(tmp_path, "w") as f:
        for r in local_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    accelerator.wait_for_everyone()

    if not is_main:
        return

    # ── merge ────────────────────────────────────
    all_rows = []
    for r in range(world):
        p = f"/tmp/eval_single_rank{r}.jsonl"
        with open(p) as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))
        os.remove(p)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── summary ──────────────────────────────────
    def avg(lst): return round(float(np.mean(lst)), 4) if lst else 0.0

    det_scores = [1.0 if r["det_correct"] else 0.0
                  for r in all_rows if r["det_correct"] is not None]
    hier_rows  = [r for r in all_rows if r["attr_f1"] is not None]
    by_ne      = defaultdict(list)
    for r in hier_rows:
        by_ne[r["num_edits"]].append(r)

    print(f"\n{'='*60}")
    print(f"  checkpoint:   {args.ckpt}")
    print(f"  n_eval:       {len(all_rows)}")
    print(f"  detection:    {avg(det_scores):.4f}")
    print(f"  attr_f1:      {avg([r['attr_f1']    for r in hier_rows]):.4f}")
    print(f"  action_acc:   {avg([r['action_acc'] for r in hier_rows]):.4f}")
    print(f"  level_acc:    {avg([r['level_acc']  for r in hier_rows]):.4f}")
    print(f"  by num_edits:")
    for ne in sorted(by_ne.keys()):
        rows = by_ne[ne]
        print(f"    ne={ne}  n={len(rows)}  "
              f"attr_f1={avg([r['attr_f1'] for r in rows]):.4f}  "
              f"level={avg([r['level_acc'] for r in rows]):.4f}")
    print(f"{'='*60}")
    print(f"\n[saved] {args.output}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
