import json

import torch
from torch.utils.data import Subset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model


class WarmupTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processor = None
        self.model = None
        self.trainer = None

    def build_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.cfg.MODEL_ID, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "right"
        return self.processor

    def build_model(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.LORA_R,
            lora_alpha=self.cfg.LORA_ALPHA,
            lora_dropout=self.cfg.LORA_DROPOUT,
            target_modules=self.cfg.LORA_TARGET_MODULES,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model

    def build_datasets(self):
        train_ds = self.cfg.SFTDataset(self.cfg.TRAIN_FILE, self.processor)
        eval_ds = self.cfg.SFTDataset(self.cfg.EVAL_FILE, self.processor)
        eval_ds = Subset(eval_ds, range(min(100, len(eval_ds))))
        print(f"train={len(train_ds)}  eval={len(eval_ds)}")

        with open(self.cfg.EVAL_FILE) as f:
            eval_records = [json.loads(line) for line in f if line.strip()]
        return train_ds, eval_ds, eval_records[:100]

    def build_training_args(self):
        return TrainingArguments(
            output_dir=self.cfg.OUTPUT_DIR,
            num_train_epochs=self.cfg.NUM_EPOCHS,
            per_device_train_batch_size=self.cfg.PER_DEVICE_BS,
            per_device_eval_batch_size=self.cfg.PER_DEVICE_BS,
            gradient_accumulation_steps=self.cfg.GRAD_ACCUM,
            learning_rate=self.cfg.LR,
            warmup_ratio=self.cfg.WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=self.cfg.LOG_STEPS,
            save_steps=self.cfg.SAVE_STEPS,
            eval_steps=self.cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_total_limit=None,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="none",
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=4,
        )

    def build(self):
        self.build_processor()
        self.build_model()
        train_ds, eval_ds, eval_records = self.build_datasets()
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        compute_metrics_fn, state_cb = self.cfg.make_compute_metrics(
            eval_records,
            self.processor,
            self.model,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.build_training_args(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=self.cfg.DataCollator(pad_id),
            compute_metrics=compute_metrics_fn,
            callbacks=[self.cfg.TrainProgressCallback(), state_cb],
        )
        return self.trainer

    def run(self):
        trainer = self.trainer or self.build()
        trainer.train()
        trainer.save_model(self.cfg.OUTPUT_DIR)
        self.processor.save_pretrained(self.cfg.OUTPUT_DIR)
        print(f"Saved to: {self.cfg.OUTPUT_DIR}")
