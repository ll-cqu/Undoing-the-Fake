import os

import torch

from undo_the_fake.configs.runtime import LORA_TARGET_MODULES


def load_processor(model_id: str, padding_side: str = "left", **kwargs):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    processor.tokenizer.padding_side = padding_side
    if hasattr(processor, "padding_side"):
        processor.padding_side = padding_side
    return processor


def load_qwen3vl(model_id: str):
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def apply_lora(model, checkpoint: str = "", r: int = 64, alpha: int = 128, dropout: float = 0.05, trainable: bool = True):
    from peft import LoraConfig, PeftModel, get_peft_model

    if checkpoint and os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        return PeftModel.from_pretrained(model, checkpoint, is_trainable=trainable)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def find_rope_index_owner(model):
    visited = set()

    def search(obj):
        if id(obj) in visited:
            return None
        visited.add(id(obj))
        if "get_rope_index" in type(obj).__dict__:
            return obj
        for attr in ("base_model", "model"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                result = search(child)
                if result is not None:
                    return result
        return None

    return search(model)


def patch_qwen3vl_forward(model):
    original_forward = model.forward

    def patched_forward(**kwargs):
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        attention_mask = kwargs.get("attention_mask")
        if mm_token_type_ids is not None and attention_mask is not None:
            seq_len = attention_mask.shape[1]
            cur_len = mm_token_type_ids.shape[1]
            if cur_len < seq_len:
                pad = torch.zeros(
                    mm_token_type_ids.shape[0],
                    seq_len - cur_len,
                    dtype=mm_token_type_ids.dtype,
                    device=mm_token_type_ids.device,
                )
                kwargs["mm_token_type_ids"] = torch.cat([pad, mm_token_type_ids], dim=1)
            elif cur_len > seq_len:
                kwargs["mm_token_type_ids"] = mm_token_type_ids[:, cur_len - seq_len:]
        return original_forward(**kwargs)

    model.forward = patched_forward
    owner = find_rope_index_owner(model)
    if owner is None:
        print("[patch] WARNING: could not find get_rope_index owner.", flush=True)
        return

    print(f"[patch] Patching get_rope_index on {type(owner).__name__}", flush=True)
    original_get_rope_index = type(owner).get_rope_index

    def patched_get_rope_index(
        self_inner,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
        position_ids=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        kwargs.pop("position_ids", None)
        if attention_mask is None or attention_mask.all():
            return original_get_rope_index(
                self_inner,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )

        bsz, seq_len = input_ids.shape
        rp_ids = input_ids.new_zeros(bsz, seq_len)
        rp_am = attention_mask.new_zeros(bsz, seq_len)
        rp_mtt = mm_token_type_ids.new_zeros(bsz, seq_len) if mm_token_type_ids is not None else None
        real_lens = []
        for batch_idx in range(bsz):
            mask_b = attention_mask[batch_idx].bool()
            real_len = int(mask_b.sum().item())
            real_lens.append(real_len)
            rp_ids[batch_idx, :real_len] = input_ids[batch_idx, mask_b]
            rp_am[batch_idx, :real_len] = 1
            if rp_mtt is not None:
                rp_mtt[batch_idx, :real_len] = mm_token_type_ids[batch_idx, mask_b]

        try:
            pos_ids, rope_deltas = original_get_rope_index(
                self_inner,
                rp_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=rp_am,
                mm_token_type_ids=rp_mtt,
                **kwargs,
            )
        except RuntimeError as exc:
            print(f"[patch] get_rope_index fallback: {exc}", flush=True)
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(3, bsz, seq_len).clone()
            rope_deltas = torch.zeros(bsz, 1, dtype=torch.long, device=input_ids.device)
            return pos_ids, rope_deltas

        lp_pos_ids = pos_ids.new_zeros(3, bsz, seq_len)
        for batch_idx, real_len in enumerate(real_lens):
            if real_len > 0:
                lp_pos_ids[:, batch_idx, seq_len - real_len:] = pos_ids[:, batch_idx, :real_len]
        return lp_pos_ids, rope_deltas

    type(owner).get_rope_index = patched_get_rope_index

