import json
import os
import random
import time

import numpy as np
import torch


class GRPOTrainingJob:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{self.local_rank}"

    def log(self, msg):
        print(f"[rank{self.rank}] {msg}", flush=True)

    def seed_everything(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def configure_output(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.cfg._LOG_OUTPUT_DIR = self.args.output_dir
        self.cfg._VIS_ENABLED = self.args.vis
        self.cfg._VIS_DIR = os.path.join(self.args.output_dir, "vis")
        if self.rank == 0:
            self.cfg._get_reward_logger(self.args.output_dir)
            if self.cfg._VIS_ENABLED:
                os.makedirs(self.cfg._VIS_DIR, exist_ok=True)
                self.log(f"Vis dir: {self.cfg._VIS_DIR}")

    def load_processor(self):
        from transformers import AutoProcessor

        self.log(">>> STEP 1: Loading processor")
        processor = AutoProcessor.from_pretrained(
            self.cfg.BASE_MODEL,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        processor.tokenizer.padding_side = "left"
        if hasattr(processor, "padding_side"):
            processor.padding_side = "left"
        return processor

    def load_model(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3VLForConditionalGeneration

        self.log(">>> STEP 2: Loading base model")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.BASE_MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=None,
            trust_remote_code=True,
        )
        model.config.use_cache = False

        self.log(">>> STEP 3: Loading LoRA from warmup checkpoint")
        adapter_config = os.path.join(self.args.checkpoint, "adapter_config.json")
        if os.path.exists(adapter_config):
            self.log("Resuming LoRA from checkpoint")
            model = PeftModel.from_pretrained(model, self.args.checkpoint, is_trainable=True)
        else:
            self.log("Fresh LoRA")
            lora_cfg = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

        model = model.to(self.device)
        model.print_trainable_parameters()
        return model

    def normalize_checkpoint(self):
        for dead_file in ["scheduler.pt", "optimizer.pt", "trainer_state.json"]:
            dead_path = os.path.join(self.args.checkpoint, dead_file)
            if os.path.exists(dead_path):
                os.rename(dead_path, dead_path + ".bak")
                self.log(f"Renamed {dead_file} to .bak")

    def patch_model(self, model):
        for module in [model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)]:
            if module is not None and not hasattr(module, "warnings_issued"):
                module.warnings_issued = {}

        self.log(">>> STEP 4: Patching Qwen3-VL")
        self.cfg._patch_qwen3vl_forward(model)

    def warmup_gan(self):
        self.log(">>> STEP 4.5: Pre-warming GAN + ArcFace")
        try:
            self.cfg.get_gan_models(self.device)
            self.cfg.get_arcface(self.device)
            self.log(f"GAN + ArcFace warmed up on {self.device}")
        except Exception as exc:
            self.log(f"WARNING: warmup failed: {exc}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def build_config(self):
        from trl import GRPOConfig

        self.log(">>> STEP 6: Building GRPOConfig")
        return GRPOConfig(
            output_dir=self.args.output_dir,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            bf16=True,
            fp16=False,
            num_generations=self.args.num_generations,
            max_completion_length=self.args.max_new_tokens,
            average_tokens_across_devices=False,
            beta=0.04,
            epsilon=0.2,
            temperature=1.0,
            top_p=0.9,
            mask_truncated_completions=True,
            max_grad_norm=1.0,
            save_steps=self.args.save_steps,
            logging_steps=self.args.logging_steps,
            save_total_limit=None,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            seed=self.args.seed,
        )

    def build_trainer(self, model, processor, dataset, grpo_config):
        from trl import GRPOTrainer

        self.log(">>> STEP 7: Building GRPOTrainer")
        trainer = GRPOTrainer(
            model=model,
            processing_class=processor,
            reward_funcs=self.cfg.compute_reward,
            args=grpo_config,
            train_dataset=dataset,
        )

        orig_call = trainer.processing_class.__call__

        def no_trunc_call(*args, **kwargs):
            kwargs.pop("max_length", None)
            kwargs.pop("truncation", None)
            return orig_call(*args, **kwargs)

        trainer.processing_class.__call__ = no_trunc_call
        return trainer

    def load_eval_records(self):
        self.log(">>> STEP 7.5: Loading eval records")
        eval_records = []
        if os.path.exists(self.args.eval_jsonl):
            with open(self.args.eval_jsonl) as f:
                eval_records = [json.loads(line) for line in f if line.strip()]
            if self.args.eval_samples:
                random.shuffle(eval_records)
                eval_records = eval_records[:self.args.eval_samples]
            self.log(f"Eval records: {len(eval_records)}")
        return eval_records

    def add_callbacks(self, trainer, model, processor):
        from transformers import TrainerCallback

        eval_records = self.load_eval_records()
        eval_cb = self.cfg.GRPOEvalCallback(
            model=model,
            processor=processor,
            eval_records=eval_records,
            output_dir=self.args.output_dir,
            eval_steps=self.args.eval_steps,
            device=self.device,
        )
        cfg = self.cfg

        class EvalAndLogCallback(TrainerCallback):
            def on_step_end(self, args_, state, control, **kwargs):
                eval_cb.on_step_end(state.global_step)

            def on_log(self, args_, state, control, logs=None, **kwargs):
                if logs is None or not cfg._LOG_OUTPUT_DIR:
                    return
                if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                    return
                entry = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "train_log",
                    "step": state.global_step,
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()},
                }
                cfg._get_reward_logger(cfg._LOG_OUTPUT_DIR).debug(json.dumps(entry, ensure_ascii=False))

        trainer.add_callback(EvalAndLogCallback())

    def run(self):
        self.seed_everything()
        self.configure_output()

        processor = self.load_processor()
        model = self.load_model()
        self.normalize_checkpoint()
        self.patch_model(model)
        self.warmup_gan()

        self.log(">>> STEP 5: Loading dataset")
        dataset = self.cfg.GRPODataset(self.args.train_jsonl)
        trainer = self.build_trainer(model, processor, dataset, self.build_config())
        self.add_callbacks(trainer, model, processor)

        self.log(">>> STEP 8: Starting training")
        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(self.args.output_dir)
        self.log(f"Done. Saved → {self.args.output_dir}")
