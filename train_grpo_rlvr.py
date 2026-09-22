"""Stage 2 -- GRPO training with a Python-verifier reward (RLVR), no reward-model weights.
See docs/PLAN.md "Stage 2 -- GRPO training" and "GPU sizing" for the reasoning behind every number; the
numbers themselves live in config.py and are shared with train_grpo_pytorch.py.

Batch-size mapping (the one part of trl's config that's easy to get wrong -- verified against trl
1.13.0's GRPOConfig, not assumed):
  num_generations               G completions per prompt
  per_device_train_batch_size   the forward/backward micro-batch of SEQUENCES (cfg.gpu.profiles:
                                2 on 24GB, 4 on 40GB)
  gradient_accumulation_steps   = FULL_BATCH_SEQUENCES / per_device_train_batch_size, so that
                                generation_batch_size (= product) is the full 64-sequence rollout
                                = 8 unique prompts x G=8 on either card. The backward pass never holds
                                more than one micro-batch's activations.

24GB vs 40GB: the profile also sets vllm_gpu_memory_utilization and vllm_enable_sleep_mode -- on 24GB trl
puts the colocated vLLM engine to sleep during the optimizer steps and wakes it for the next rollout.

Checkpoints: trl/transformers save every cfg.checkpoint.every steps to OUTPUT_DIR/checkpoint-N (adapter,
optimizer, scheduler, RNG), keep cfg.checkpoint.keep_last, and cfg.checkpoint.resume="auto" continues from the
newest one.

Run: `python train_grpo_rlvr.py`. Needs a GPU. smoke / gpu.profile / checkpoint.resume are set in config.py.
"""
import os

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from config import GPUProfile, cfg
from data.dataset import load_grpo_datasets
from reward_fn import active_rewards


def build_args(profile: GPUProfile) -> GRPOConfig:
    micro = profile.micro_batch_sequences
    return GRPOConfig(
        output_dir=cfg.checkpoint.output_dir_rlvr,
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=cfg.model.bf16,
        seed=cfg.optim.seed,
        num_generations=cfg.batch.num_generations,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=cfg.batch.full_batch_sequences // micro,
        max_completion_length=cfg.batch.max_completion_length,
        reward_weights=active_rewards()[1],
        temperature=cfg.batch.temperature,
        top_p=1.0,
        top_k=0,                # plain softmax; the model's generation_config ships top_k=20/top_p=0.8
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        warmup_steps=cfg.optim.warmup_steps(),  # warmup_ratio is deprecated in this transformers version
        lr_scheduler_type={"wsd": "warmup_stable_decay", "linear": "linear", "constant": "constant_with_warmup"}[cfg.optim.schedule],
        lr_scheduler_kwargs={"num_decay_steps": int(cfg.optim.max_steps * cfg.optim.decay_frac)} if cfg.optim.schedule == "wsd" else None,
        max_grad_norm=cfg.optim.grad_clip_norm,
        beta=cfg.optim.beta,          # 0 = no reference model -- DAPO/Dr.GRPO default, already trl's default
        num_iterations=cfg.optim.num_iterations,  # mu: 1 => ratio is always 1, clip inactive
        epsilon=cfg.optim.clip_eps,
        disable_dropout=True,   # same reason as lora_dropout=0: the ratio must see weight movement only
        use_vllm=True,
        vllm_mode="colocate",
        # trl detects the bitsandbytes 4-bit base and starts vLLM with quantization="bitsandbytes". vLLM 0.28
        # dropped that loader ("Unknown quantization method: bitsandbytes", seen 2026-09-18 with the PyTorch
        # script), so this path is expected to fail on the pinned vLLM until trl is told to serve bf16 weights;
        # train_grpo_pytorch.py is the working path. Without vllm_max_model_length vLLM plans for the model's
        # native 262K context and refuses to start when the KV cache can't hold one such sequence.
        vllm_max_model_length=cfg.batch.vllm_max_model_len,
        vllm_gpu_memory_utilization=profile.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=profile.vllm_sleep_mode,
        max_steps=cfg.optim.max_steps,
        logging_steps=1,
        save_steps=cfg.checkpoint.every,
        save_total_limit=cfg.checkpoint.keep_last,
        # No in-loop eval: with num_generations_eval defaulting to 8 it would generate 500x8 completions
        # every eval_steps -- about as much compute as 60 training steps. evaluate.py is the held-out measurement.
        eval_strategy="no",
        report_to="none",       # set to "wandb"/"tensorboard" for run tracking
    )


def build_trainer(profile: GPUProfile) -> GRPOTrainer:
    quantization_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16, **cfg.model.bnb_4bit)
    peft_config = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,  # RL: the importance ratio must see weight movement only, not dropout noise
        task_type="CAUSAL_LM",
    )
    train_dataset, _ = load_grpo_datasets(cfg.data_dir)
    return GRPOTrainer(
        model=cfg.model.model_id,
        reward_funcs=active_rewards()[0],
        args=build_args(profile),
        train_dataset=train_dataset,
        quantization_config=quantization_config,
        peft_config=peft_config,
    )


def find_resume_checkpoint() -> str | None:
    """cfg.checkpoint.resume: "no" -> None; a path -> that path; "auto" -> newest OUTPUT_DIR/checkpoint-N if any."""
    if cfg.checkpoint.resume in ("no", "0", "false", "False"):
        return None
    if cfg.checkpoint.resume != "auto":
        if not os.path.isdir(cfg.checkpoint.resume):
            raise SystemExit(f"checkpoint.resume={cfg.checkpoint.resume!r} is not a directory")
        return cfg.checkpoint.resume
    return get_last_checkpoint(cfg.checkpoint.output_dir_rlvr) if os.path.isdir(cfg.checkpoint.output_dir_rlvr) else None


def main() -> None:
    os.makedirs(cfg.checkpoint.output_dir_rlvr, exist_ok=True)
    profile = cfg.resolve_gpu_profile()
    print(f"config:\n{cfg.summary()}\n  gpu_profile              = {profile}")
    trainer = build_trainer(profile)
    resume = find_resume_checkpoint()
    if resume:
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(f"{cfg.checkpoint.output_dir_rlvr}/final")


if __name__ == "__main__":
    main()
