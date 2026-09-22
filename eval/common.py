"""Shared model loading + batched generation/scoring for pilot_baseline.py, rerun_truncated.py and evaluate.py.

Two inference backends behind one loop:
  hf    -- transformers model.generate() on an NF4 (bitsandbytes) base, the original path. Slow: no paged
           KV cache, a batch of 8 at a 4096 cap is minutes per batch.
  vllm  -- vLLM engine, same model, bf16 weights (~8GB; vLLM 0.28 dropped the bitsandbytes loader, so --quant bnb
           only works on older versions). LoRA checkpoints are served through vLLM's LoRA support, no merge.

Before/after comparisons must use the SAME backend and quant on both sides: switching hf-nf4 -> vllm-bf16
between baseline and after would credit the quantization change to training. Every result row records
its backend so a mismatch is visible in the JSONL.
"""
import argparse
import os
import time

import torch
from tqdm.auto import tqdm

from reward_fn import math_reward

from config import cfg

MODEL_ID = cfg.model.model_id  # same model / LoRA rank as training (config.py)
LORA_RANK = cfg.lora.r


def log(msg: str) -> None:
    """Timestamped status line. Goes through tqdm.write so it does not corrupt a live progress bar."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Backend: transformers
# ---------------------------------------------------------------------------
def load_model(adapter_path: str | None = None):
    """NF4-quantized base, matching training's quantization (docs/PLAN.md Model section) so any
    before/after difference reflects the LoRA training, not a quantization change. Pass
    adapter_path to attach a trained checkpoint for "after" eval; None gives the untouched
    base model for the Stage 1 baseline."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log(f"loading tokenizer {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log("loading base model in NF4 (first run downloads ~8GB of weights, then quantizes on load; "
        "several minutes with no output is normal)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config, dtype=torch.bfloat16, device_map="auto",
    )
    log(f"base model ready on {model.device} in {time.time() - t0:.0f}s")
    if adapter_path is not None:
        log(f"attaching LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


class HFEngine:
    """model.generate() wrapper. Greedy, left-padded batch, returns (completion_text, completion_tokens)."""
    name = "hf-nf4"

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def generate_texts(self, texts: list[str], max_new_tokens: int) -> list[tuple[str, int]]:
        tok = self.tokenizer
        enc = tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
        )
        completion_ids = out[:, prompt_len:]
        completions = tok.batch_decode(completion_ids, skip_special_tokens=True)
        lengths = (completion_ids != tok.pad_token_id).sum(dim=-1).tolist()
        return list(zip(completions, lengths))


# ---------------------------------------------------------------------------
# Backend: vLLM
# ---------------------------------------------------------------------------
class VLLMEngine:
    """vLLM offline engine. The adapter, if any, is attached per request via LoRARequest, so the base
    weights stay untouched and the same engine could serve base and adapter side by side."""

    def __init__(self, adapter_path: str | None, quant: str, max_model_len: int, gpu_memory_utilization: float):
        # vLLM defaults to FlashInfer's top-k/top-p sampler, which JIT-compiles with nvcc on first use and fails
        # on images that ship nvcc without the CUDA headers (SageMaker Distribution: "cuda_runtime.h: No such
        # file"). We decode greedily, so the fused sampler gains nothing; use vLLM's PyTorch sampler instead.
        # Must be set before the engine subprocess is spawned, which inherits this environment.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        from train_grpo_pytorch import check_vllm_quant
        check_vllm_quant(quant)
        self._SamplingParams = SamplingParams
        self.name = f"vllm-{'nf4' if quant == 'bnb' else 'bf16'}"
        kwargs = dict(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
            max_lora_rank=LORA_RANK if adapter_path is not None else None,
            seed=0,
        )
        if quant == "bnb":
            kwargs["quantization"] = "bitsandbytes"  # in-flight NF4, same family as training's BitsAndBytesConfig
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        log(f"starting vLLM: quant={quant} max_model_len={max_model_len} "
            f"gpu_memory_utilization={gpu_memory_utilization} lora={'yes' if adapter_path else 'no'} "
            "(engine startup compiles graphs and profiles memory; a minute or two of vLLM's own logs is normal)")
        t0 = time.time()
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = LoRARequest("after", 1, adapter_path) if adapter_path else None
        if adapter_path:
            log(f"LoRA adapter registered from {adapter_path}")
        log(f"vLLM ready in {time.time() - t0:.0f}s")

    def generate_texts(self, texts: list[str], max_new_tokens: int) -> list[tuple[str, int]]:
        params = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=0)
        outputs = self.llm.generate(texts, params, lora_request=self.lora_request, use_tqdm=False)
        # vLLM returns outputs in input order; text already has special tokens stripped.
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]

    def sample_texts(self, texts: list[str], max_new_tokens: int, n: int, temperature: float,
                     seed: int = 0) -> list[list[tuple[str, int]]]:
        """n samples per prompt from the tempered softmax (no top-p/top-k truncation, as in training)."""
        params = self._SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=0, min_p=0.0,
                                      max_tokens=max_new_tokens, seed=seed)
        outputs = self.llm.generate(texts, params, lora_request=self.lora_request, use_tqdm=False)
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outputs]


# ---------------------------------------------------------------------------
# CLI plumbing shared by the three scripts
# ---------------------------------------------------------------------------
PROMPT_HEADROOM = 1024  # longest holdout prompt is 626 tokens (2026-09-18 analysis); leave margin


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="inference engine. vllm is the fast path; hf is the original transformers path")
    ap.add_argument("--quant", choices=["bnb", "bf16"], default="bf16",
                    help="vllm only: bf16 = full weights (~8GB, default). bnb = in-flight NF4, only on vLLM versions "
                         "that still ship the bitsandbytes loader (0.28 does not)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="prompts per generate() call. Default 8 for hf, 64 for vllm (vLLM schedules within "
                         "the chunk, so bigger is faster; the chunk only sets how often the log line updates)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9, help="vllm only: fraction of GPU memory to claim")


def resolve_batch_size(args) -> int:
    if args.batch_size is not None:
        return args.batch_size
    return 64 if args.backend == "vllm" else 8


def load_engine(args, adapter_path: str | None = None):
    """Build the engine named by --backend. Returns (engine, tokenizer)."""
    if args.backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            raise SystemExit("vllm is not installed. `pip install 'vllm>=0.19.1,<=0.28.0'` (see requirements.txt) "
                             "or rerun with --backend hf")
        engine = VLLMEngine(
            adapter_path, quant=args.quant, max_model_len=args.max_new_tokens + PROMPT_HEADROOM,
            gpu_memory_utilization=args.gpu_mem_util,
        )
        return engine, engine.tokenizer
    model, tokenizer = load_model(adapter_path=adapter_path)
    return HFEngine(model, tokenizer), tokenizer


# ---------------------------------------------------------------------------
# The shared loop
# ---------------------------------------------------------------------------
def generate_and_score(engine, tokenizer, dataset, batch_size: int = 8, max_new_tokens: int = 2048,
                       desc: str = "eval"):
    """Greedy-decode one completion per example (deterministic, so before/after and repeat
    runs are comparable -- training samples at temperature=1.0, but eval doesn't need to match
    that, it needs to be reproducible). Returns a list of per-example result dicts.
    Shows a tqdm bar over batches with the running pass rate, since a full holdout at 2048
    tokens per completion runs for a long time with no other output.
    `engine` is an HFEngine/VLLMEngine; a bare transformers model is accepted and wrapped."""
    if not hasattr(engine, "generate_texts"):
        engine = HFEngine(engine, tokenizer)
    results = []
    num_correct = 0
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    log(f"{desc}: {len(dataset)} examples, {num_batches} batches of {batch_size}, "
        f"greedy up to {max_new_tokens} new tokens each, backend {engine.name}")
    progress = tqdm(range(0, len(dataset), batch_size), desc=desc, unit="batch")
    for batch_num, start in enumerate(progress, start=1):
        batch = dataset[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in batch["prompt"]
        ]
        if batch_num == 1:
            log("generating batch 1 (the first batch also warms up kernels, so it is the slowest; "
                "a batch that hits the token limit can take a few minutes)")
        t0 = time.time()
        generated = engine.generate_texts(texts, max_new_tokens)
        gen_seconds = time.time() - t0
        completions = [text for text, _ in generated]
        lengths = [n for _, n in generated]
        rewards = math_reward(completions, batch["ground_truth"])

        for i in range(len(completions)):
            results.append({
                "problem_id": batch["problem_id"][i],
                "source": batch["source"][i],
                "ground_truth": batch["ground_truth"][i],
                "completion": completions[i],
                "reward": rewards[i],
                "completion_tokens": lengths[i],
                "backend": engine.name,
            })
        num_correct += sum(rewards)
        progress.set_postfix(pass_rate=f"{num_correct / len(results):.3f}", n=len(results))
        log(f"batch {batch_num}/{num_batches}: {gen_seconds:.0f}s, "
            f"{sum(rewards):.0f}/{len(rewards)} correct, max {max(lengths)} tokens, "
            f"running pass rate {num_correct / len(results):.3f}")
    return results
