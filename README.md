# Undo the Fake

## Installation

Create a Python environment and install dependencies:

```bash
python3 -m venv undo_fake
source undo_fake/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Preparation

### Dataset

Prepare CelebA-HQ Mask with the following expected layout:

```text
data/CelebAMask-HQ/
  CelebA-HQ-img/
  CelebAMask-HQ-attribute-anno.txt
```

The data generation pipeline reads the CelebA-HQ images and the CelebA-HQ attribute annotation file, samples attribute edit plans, and uses StyleCLIP to generate edited images plus metadata.

After the dataset and StyleCLIP assets are prepared, generate the edited dataset with:

```bash
python generate_dataset.py \
  --celeba_img_dir data/CelebAMask-HQ/CelebA-HQ-img \
  --celeba_anno data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt \
  --plan_json data/edit_plans.json \
  --results_json data/edit_results.jsonl \
  --failed_json data/edit_failed.jsonl \
  --edited_dir data/styleclip_edited \
  --num_gpus 8 \
  --variants 3 \
  --batch_size 16
```

The command writes edit plans to `data/edit_plans.json`, successful generated samples to `data/edit_results.jsonl`, failed samples to `data/edit_failed.jsonl`, and generated images to `data/styleclip_edited/`.

### Models

Prepare the base VLM:

```text
hf_models/Qwen3-VL-8B-Instruct/
```

You can also point to a Hugging Face model path or local model path with:

```bash
export UTF_BASE_MODEL=hf_models/Qwen3-VL-8B-Instruct
```

Prepare StyleCLIP and related GAN assets:

```text
StyleCLIP/
  global_torch/
    model/ffhq.pkl
    npy/ffhq/fs3.npy

face_editing/
  encoder4editing/
  pretrained/e4e_ffhq_encode.pt
```


## Warmup Training

Run supervised warmup with Qwen3-VL-8B LoRA:

```bash
accelerate launch --num_processes 8 warmup.py
```

Warmup defaults are defined in:

```text
undo_the_fake/training/warmup.py
```

The default inputs are:

```text
data/warmup/warmup_train.jsonl
data/warmup/warmup_test.jsonl
```

The default output is:

```text
output_file/warmup
```

## GRPO Training

Run GRPO training from the warmup checkpoint:

```bash
accelerate launch --num_processes 8 train.py \
  --train_jsonl data/grpo/grpo_gan/grpo_train.jsonl \
  --checkpoint output_file/warmup_checkpoint \
  --output_dir output_file/train \
  --batch_size 1 \
  --grad_accum 4 \
  --num_generations 4
```

Useful options:

```bash
--eval_jsonl data/warmup/warmup_test.jsonl
```

The GRPO trainer logs reward components and optional recovery visualizations under the selected `--output_dir`.

## Evaluation

Evaluate a checkpoint and write per-sample predictions:

```bash
accelerate launch --num_processes 8 eval_grpo_gan.py \
  --ckpt output_file/train_checkpoint \
  --eval_jsonl data/warmup/warmup_test.jsonl \
  --output output_file/eval_samples.jsonl \
  --eval_samples 0
```

`--eval_samples 0` evaluates the full file. Use a positive number for a sampled subset.

## Metric Calculation

Calculate detection, localization, level, and per-attribute metrics:

```bash
python calculate_metric.py \
  --input output_file/eval_samples.jsonl \
  --output_dir output_file/metrics \
  --meta_jsonl data/warmup/warmup_test.jsonl
```

To include recovery metrics with StyleCLIP and ArcFace:

```bash
accelerate launch --num_processes 8 calculate_metric.py \
  --input output_file/eval_samples.jsonl \
  --output_dir output_file/metrics_recovery \
  --meta_jsonl data/warmup/warmup_test.jsonl \
  --recovery \
```

Metric outputs include:

```text
analysis_summary.json
all_metrics.jsonl
recovery_failed.jsonl  # only when recovery is enabled and failures/skips exist
```

