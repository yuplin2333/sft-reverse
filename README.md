# sft-reverse

Code for reversible SFT behavior experiments (LCDD + SFT-Eraser).

## Scope

This repository provides the code for the core pipeline:

1. Two-phase training (`Full SFT` -> `LCDD`)
2. Soft-trigger optimization (`SFT-Eraser`)
3. Main evaluations (task metrics + KL benchmark + utility benchmark)

## Environment

- Python >= 3.11
- `uv` package manager
- CUDA GPU recommended for training/evaluation

Install dependencies:

```bash
uv sync
```

## Data

Most datasets are loaded directly from Hugging Face in the scripts:

- `tatsu-lab/alpaca` (Fixed Response task)
- `allenai/wildjailbreak` (Safety task)
- `Roudranil/shakespearean-and-modern-english-conversational-dataset` (Shakespeare task)

For safety evaluation, the HarmBench behavior CSV is expected at:

- `data/harmbench/harmbench_behaviors_text_all.csv`

If missing, download it from the HarmBench release and place it at the path above.

## Reproduction Commands

All commands use `uv run`.

### 1) Train (Full SFT or LCDD)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --task safety \
  --base-model Qwen/Qwen3-0.6B \
  --num-samples 5000 \
  --num-epochs 3 \
  --learning-rate 2e-5 \
  --output-dir checkpoints/example_sft \
  --no-l0
```

LCDD phase example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --task safety \
  --base-model checkpoints/example_sft/final \
  --l0-base-model Qwen/Qwen3-0.6B \
  --loss-budget-ratio 0.30 \
  --l0-lambda-init 0.01 \
  --lambda-lr 0.1 \
  --l0-mask-lr 0.1 \
  --l0-warmup-steps 300 \
  --mask-optimizer sgd \
  --mask-momentum 0.0 \
  --num-samples 5000 \
  --num-epochs 20 \
  --learning-rate 2e-5 \
  --output-dir checkpoints/example_lcdd
```

### 2) Optimize Trigger

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/optimize_trigger.py \
  --task safety \
  --model checkpoints/example_lcdd/final \
  --base-model Qwen/Qwen3-0.6B \
  --method statistical_kl \
  --channel-type write \
  --trigger-length 20 \
  --batch-size 16 \
  --lr 0.003 \
  --kl-weight 0.7 \
  --kl-tail-k 8 \
  --l2 0.1 \
  --max-norm 1.0 \
  --n-pairs 200 \
  --n-steps 2000 \
  --output-dir results/triggers/example
```

### 3) Evaluate

Safety benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_safety_bench.py \
  --model checkpoints/example_lcdd/final \
  --num-samples 200 \
  --output-dir results/benchmarks/safety/example
```

KL benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_kl_bench.py \
  --task safety \
  --base-model Qwen/Qwen3-0.6B \
  --sft-model checkpoints/example_sft/final \
  --l0-model checkpoints/example_lcdd/final \
  --trigger results/triggers/example/trigger_embeds.pt \
  --num-samples 200 \
  --output-dir results/kl/example
```

Utility benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_utility_bench.py \
  --model checkpoints/example_lcdd/final \
  --output-dir results/benchmarks/utility/example
```

## Task-Specific Examples

### Fixed Response (`idk`)

Train (Full SFT):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --task idk \
  --base-model Qwen/Qwen3-0.6B \
  --num-samples 5000 \
  --num-epochs 3 \
  --learning-rate 2e-5 \
  --output-dir checkpoints/idk_example_sft \
  --no-l0
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_idk_bench.py \
  --model checkpoints/idk_example_sft/final \
  --num-samples 200 \
  --output-dir results/benchmarks/idk/example
```

### Shakespeare (`shakespeare`)

Train (Full SFT):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --task shakespeare \
  --base-model Qwen/Qwen3-0.6B \
  --num-samples 5000 \
  --num-epochs 3 \
  --learning-rate 2e-5 \
  --output-dir checkpoints/shakespeare_example_sft \
  --no-l0
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_shakespeare_bench.py \
  --model checkpoints/shakespeare_example_sft/final \
  --num-samples 200 \
  --output-dir results/benchmarks/shakespeare/example
```
