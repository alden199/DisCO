# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DisCO (Discriminative Constrained Optimization) is a reinforcement learning framework for training large reasoning models. It implements a novel discriminative learning approach as an alternative to GRPO, addressing issues like difficulty bias and entropy collapse while enabling faster convergence.

**Key papers:** NeurIPS 2025 (arXiv:2505.12366) and ICLR 2026 follow-up.

## Installation

```bash
conda create -n disco python=3.10
conda activate disco

# Install packages in development mode
pip install -e ./verl
pip install -e ./deepscaler
pip install wandb

# If vLLM conflicts occur:
pip install --no-deps vllm==0.6.3
pip install outlines==0.0.6 xformers==0.0.27.post2 torchvision==0.19 torch==2.4.0
pip uninstall -y vllm-flash-attn
```

## Training Commands

```bash
# Single-node (8x A100-80GB GPUs)
bash ./scripts/train/run_disco_logL_1.5b_8k.sh

# Multi-node: start Ray cluster first
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head  # on head node
ray start --address=[RAY_ADDRESS]  # on worker nodes
bash ./scripts/train/run_disco_logL_7b_8k.sh
```

Available training variants in `scripts/train/`:
- `run_disco_logL_*.sh` - DisCO with log-likelihood scoring
- `run_disco_Lratio_*.sh` - DisCO with likelihood ratio scoring
- `run_discob_*.sh` - Balanced DisCO variants

## Evaluation

```bash
./scripts/eval/eval_model.sh --model [CHECKPOINT_PATH] --datasets aime math minerva --output-dir ./val_results/
```

## Architecture

### Training Pipeline

```
main_ppo.py → RayPPOTrainer
    ├── ActorRolloutRefWorker (policy, generation, reference)
    ├── CriticWorker (value model)
    ├── RewardModelWorker (optional)
    └── RewardManager → deepscaler_reward_fn (math evaluation)
```

### Core Components

**verl/trainer/ppo/core_algos.py** - RL algorithm implementations:
- `compute_policy_loss_disco_logL` / `compute_policy_loss_disco_Lratio` - DisCO variants
- `compute_policy_loss_discob_*` - Balanced DisCO
- `compute_policy_loss_grpo` - GRPO baseline
- `compute_gae_advantage_return` - GAE for PPO

**verl/trainer/ppo/ray_trainer.py** - Distributed training orchestration via Ray

**verl/workers/** - Distributed workers:
- `fsdp_workers.py` - FSDP-based training
- `megatron_workers.py` - Megatron-LM support
- `rollout/vllm_rollout.py` - vLLM generation backend

**deepscaler/rewards/math_reward.py** - Math problem reward computation (answer extraction, symbolic/numerical comparison)

**verl/protocol.py** - DataProto: unified data structure for tensors and metadata

### Configuration (Hydra)

Main config: `verl/verl/trainer/config/ppo_trainer.yaml`

Key parameters:
- `loss_type`: RL, disco_logL, disco_Lratio, discob_*
- `adv_estimator`: gae or grpo
- `delta`, `beta`, `tau`: DisCO hyperparameters
- `ppo_max_token_len_per_gpu`: memory optimization

Override via CLI:
```bash
python main_ppo.py algorithm.adv_estimator=grpo actor_rollout_ref.actor.loss_type=disco_logL
```

## Environment Variables

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS  # Required for vLLM stability
export TOKENIZERS_PARALLELISM=true
```

## Pre-trained Checkpoints

Available on HuggingFace:
- `ganglii/DisCO-1.5B-logL`, `ganglii/DisCO-1.5B-Lratio`
- `ganglii/DisCO-7B-logL`, `ganglii/DisCO-7B-Lratio`

Base models: DeepSeek-R1-Distill-Qwen series
