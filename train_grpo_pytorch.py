"""Stage 2, pure PyTorch/HF-transformers version -- no trl.GRPOTrainer.

Same model, data, reward, and hyperparameters as train_grpo_rlvr.py (all from config.py). The difference is
that every step of the algorithm is written out explicitly here instead of living inside a library
Trainer -- useful for actually seeing the mechanism.

Rollouts (cfg.rollout.backend): "vllm" (default) runs a colocated vLLM engine on the same GPU, serving the
same base model with the live LoRA adapter hot-swapped in every step (adapter saved to disk, loaded via
LoRARequest -- no weight merge, no second training copy). This is what makes 64 concurrent 4096-token
rollouts fit: vLLM's paged KV cache and continuous batching vs model.generate()'s lock-stepped, padded
batch. "hf" keeps the plain model.generate() rollout, slower but with nothing hidden. The training side
(log-probs, loss, optimizer) is identical for both; only where the sampled tokens come from differs.

24GB vs 40GB (cfg.gpu.profiles, auto-detected): on 40GB the engine and the training model share the card
(vLLM claims 0.5 of it up front). On 24GB the two phases can't both hold their peak, so vLLM is put to
sleep (level 1: KV cache freed, weights parked in CPU RAM) during the gradient steps and woken before the
next rollout; torch.cuda.empty_cache() before wake_up() hands the allocator's cached blocks back, or the wake
OOMs. Level 2 would discard the weights too and needs an explicit reload on wake (see VLLMRollout.wake).

Adaptive sampling (cfg.batch.max_rollouts_per_prompt / park_steps / park_after_all_correct): a prompt whose G
rollouts all score 0 is re-sampled in rounds of G until a success appears or the budget is spent; its group is
then the successes plus a random fill of failures, still G wide. Prompts that exhaust the budget, or that the
policy solves G/G twice in a row with no length variance, are parked for park_steps. See collect_groups().

Checkpoints (cfg.checkpoint.every): adapter + optimizer + LR scheduler + RNG state + step + parked prompts, under
OUTPUT_DIR/step_N, with OUTPUT_DIR/latest naming the newest. cfg.checkpoint.resume="auto" continues from it;
because RNG state is restored, the resumed run draws the same prompts it would have drawn.

grpo_advantages() and grpo_loss() follow DeepSeekMath's GRPO (Shao et al. 2024, Algorithm 1) as written up in
the author's post-training study notes (not part of this repo): same formulas, same variable names, verified
against that write-up's doctest values. One deliberate deviation from the paper's prose formula: the loss
normalizes by total completion tokens in the rollout batch, not by each response's own length (1/|y_i|) --
the Dr.GRPO/DAPO convention, and trl's default loss_type="dapo". beta=0 throughout (docs/PLAN.md Stage 2), so
there is no reference model and no KL term -- policy and reward are the only things in memory.

pi_old vs pi_theta (DeepSeekMath's GRPO, Algorithm 1): both are the SAME computation -- feed
(question, response-so-far) through the policy, softmax over the vocab, read off the entry for
the token that was actually sampled -- at two different moments. pi_old is computed once per
rollout batch, right after generation, with the weights that generated it, and STORED
(logp_old below). pi_theta is recomputed from the live weights on every gradient step. DeepSeek
runs mu gradient steps per batch (NUM_ITERATIONS); on step 1 the weights haven't moved yet so
the ratio is exactly 1, on steps 2..mu it isn't, and that is when the clip matters.
Dropout is disabled (lora_dropout=0): the ratio must compare one deterministic function at two
weight states, and dropout adds noise to it that is not policy drift.

Sampling is a tempered softmax (cfg.batch.temperature, no top_p/top_k truncation) and the scorer uses the
SAME temperature: sequence_logprobs divides the logits by T before the log-softmax, so logp_old and logp
describe exactly the distribution the tokens were drawn from. The model's generation_config.json ships
top_k=20, top_p=0.8; if those leaked into generation, the tokens would come from a truncated distribution
the scorer can't express, and the importance ratio would compare two different policies.

Run from the project root: `python train_grpo_pytorch.py`. Needs a GPU. Everything is set in config.py:
smoke (few-minute run, every code path), gpu.profile (auto | 24gb | 40gb), checkpoint.resume (auto | no |
<dir>), rollout.backend (vllm | hf).
"""
import gc
import json
import os
import shutil
import sys
import time

# PyTorch's caching allocator fragments under this workload (micro-batches of changing shapes every step): the
# 2026-09-20 OOM had 3.2GB reserved-but-unallocated. Expandable segments let it grow blocks instead of holding
# stranded ones. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import cfg
from data.dataset import eval_subset, load_grpo_datasets
from reward_fn import total_reward


# ---------------------------------------------------------------------------
# GRPO math -- DeepSeekMath GRPO, Algorithm 1
# ---------------------------------------------------------------------------

def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards: [num_prompts, G] -> advantages normalized WITHIN each prompt's group.
    Population std (unbiased=False): the G rollouts are the entire comparison set, not a
    sample of a larger population. std=0 (all G rollouts scored the same) -> all-zero
    advantages -> that prompt contributes no gradient this step (the degenerate case)."""
    mu = rewards.mean(dim=-1, keepdim=True)
    sd = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - mu) / (sd + eps)


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
              clip: float | None = None, normalizer: torch.Tensor | None = None) -> torch.Tensor:
    """logp, logp_old, mask: [batch, completion_len]. adv: [batch], per-SEQUENCE, broadcast to
    every token of that sequence -- GRPO does no per-token credit assignment.
    normalizer: total completion tokens of the whole rollout batch, so that summing this loss
    over micro-batches gives exact batch-token normalization; defaults to this chunk's tokens."""
    if clip is None:
        clip = cfg.optim.clip_eps
    ratio = torch.exp(logp - logp_old)
    a = adv.unsqueeze(-1)
    obj = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    if normalizer is None:
        normalizer = mask.sum().clamp(min=1)
    return -(obj * mask).sum() / normalizer


LOGPROB_CHUNK_TOKENS = 1024  # batch x positions per log-softmax slab (= the old 2 x 512); bounds the fp32 vocab
                             # transient at ~1024 x 151,936 x 4B = 0.6GB (+ its backward copies) REGARDLESS of how
                             # many sequences the packer put in the micro-batch. Sizing by positions alone let an
                             # 8-sequence micro-batch of short answers allocate 4x the transient and OOM (2026-09-20).


def _chunk_logprobs(logits_chunk: torch.Tensor, targets_chunk: torch.Tensor, temperature: float) -> torch.Tensor:
    """log_softmax(x/T)[target] == x[target]/T - logsumexp(x/T), computed in fp32 for one slab of positions."""
    x = logits_chunk.float() / temperature
    return x.gather(dim=-1, index=targets_chunk.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(x, dim=-1)


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_len: int,
                      temperature: float | None = None) -> torch.Tensor:
    """log pi_T(o_t | q, o_<t) for each of the last `completion_len` tokens -> [batch, completion_len], under the
    tempered policy softmax(logits / T) that the rollout sampled from (T = cfg.batch.temperature by default).
    logits_to_keep asks the model for only the positions that predict the completion (the prompt's
    logits are never materialized). The vocab-wide log-softmax is the memory hotspot: at 4096
    completion tokens it is batch x 4097 x 151,936 x 4B = 2.5GB per sequence in fp32, and autograd would
    keep all of it for backward. So it is done in slabs of LOGPROB_CHUNK_TOKENS // batch positions, each
    under torch.utils.checkpoint: the forward keeps only the bf16 logits slab (a view, no copy) and
    the fp32 upcast + logsumexp are recomputed during backward. Same numbers as
    log_softmax(logits.float()).gather(targets); peak extra memory ~ 1024 x vocab x 4B = 0.6GB per slab
    whatever the batch size, instead of batch x 2.5GB."""
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=completion_len + 1, use_cache=False,
    ).logits[:, :-1]
    targets = input_ids[:, -completion_len:]
    if temperature is None:
        temperature = cfg.batch.temperature
    out = []
    positions_per_slab = max(1, LOGPROB_CHUNK_TOKENS // input_ids.shape[0])
    for s in range(0, completion_len, positions_per_slab):
        e = min(s + positions_per_slab, completion_len)
        if torch.is_grad_enabled():
            out.append(torch.utils.checkpoint.checkpoint(
                _chunk_logprobs, logits[:, s:e], targets[:, s:e], temperature, use_reentrant=False,
            ))
        else:
            out.append(_chunk_logprobs(logits[:, s:e], targets[:, s:e], temperature))
    return torch.cat(out, dim=1)


def completion_mask_from_ids(completion_ids: torch.Tensor, eos_token_ids: list[int]) -> torch.Tensor:
    """1 for every token up to and including the first EOS, 0 after. generate() right-pads each
    row after its EOS with pad_token_id, and for this tokenizer pad ("<|endoftext|>") is itself one
    of the two EOS ids (generation_config: [151645, 151643]), so "!= pad" is not a safe mask."""
    is_eos = torch.isin(completion_ids, torch.tensor(eos_token_ids, device=completion_ids.device))
    eos_before = is_eos.cumsum(-1) - is_eos.int()  # number of EOS tokens strictly before each position
    return (eos_before == 0).to(torch.bfloat16)


def make_lr_scheduler(optimizer, warmup: int, total: int, schedule: str | None = None,
                      decay_frac: float | None = None) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup from lr/warmup (not 0: HF's lambda(0) = 0 wastes the first optimizer step, ~17 min on the A10G), then
    one of (cfg.optim.schedule):
      wsd       constant, then linear decay to 0 over the last decay_frac of `total`. Every step before the decay
                trains at full rate, and extending max_steps on resume only moves the decay, it does not reshape
                what already ran. The default for a run whose length is decided by watching the curve.
      linear    linear decay to 0 at `total` (transformers' get_linear_schedule_with_warmup shape).
      constant  flat after warmup."""
    schedule = schedule or cfg.optim.schedule
    decay_frac = cfg.optim.decay_frac if decay_frac is None else decay_frac
    decay_start = int(total * (1 - decay_frac))

    def lam(step):  # step = number of completed optimizer steps
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return max(0.0, (total - step) / max(1, total - warmup))
        if schedule == "wsd":
            if step < decay_start:
                return 1.0
            return max(0.0, (total - step) / max(1, total - decay_start))
        raise SystemExit(f"unknown optim.schedule {schedule!r}: wsd | linear | constant")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lam)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_base_model():
    """NF4 base with gradient checkpointing, no adapter yet."""
    quant_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16, **cfg.model.bnb_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id, quantization_config=quant_config, dtype=torch.bfloat16, device_map="auto",
    )
    # Not peft.prepare_model_for_kbit_training: it upcasts every non-quantized parameter to fp32,
    # which for this model includes the tied 389M-parameter embedding / LM head -- that doubles its
    # memory and makes every logits tensor fp32. Gradient checkpointing with use_reentrant=False
    # is the only part of it we need.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def attach_lora(base, adapter_dir: str | None):
    """Fresh LoRA (adapter_dir=None) or the trainable adapter from a checkpoint. peft keeps the
    adapters in fp32: the "master weights"."""
    if adapter_dir is None:
        peft_config = LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout, task_type="CAUSAL_LM",
        )
        return get_peft_model(base, peft_config)
    return PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    tokenizer.padding_side = "left"  # required for batched generation
    return tokenizer


# ---------------------------------------------------------------------------
# Rollout: transformers
# ---------------------------------------------------------------------------

def tokenize_prompts(tokenizer, prompts: list) -> list[list[int]]:
    """Conversational prompts -> chat-templated token ids, once. The same ids go to the sampler (vLLM as
    prompt_token_ids, or the HF path below) and into the training forward pass."""
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


@torch.no_grad()
def hf_sample(model, tokenizer, prompt_ids: list[list[int]], n: int, max_tokens: int | None = None,
              greedy: bool = False) -> list[list[list[int]]]:
    """Plain model.generate() sampler: per prompt, n completions as token-id lists (up to and including the
    first EOS; exactly max_tokens when the cap is hit). Same contract as VLLMRollout.generate.
    greedy=True is the eval mode: deterministic argmax decoding, n must be 1."""
    rows = [ids for ids in prompt_ids for _ in range(n)]  # repeat_interleave order
    pad, prompt_len = tokenizer.pad_token_id, max(len(r) for r in rows)
    input_ids = torch.full((len(rows), prompt_len), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, r in enumerate(rows):  # left-pad
        input_ids[i, prompt_len - len(r):] = torch.tensor(r)
        attention_mask[i, prompt_len - len(r):] = 1
    out = model.generate(
        input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
        max_new_tokens=max_tokens or cfg.batch.max_completion_length,
        do_sample=not greedy,
        temperature=None if greedy else cfg.batch.temperature,
        top_p=None if greedy else 1.0,
        top_k=None if greedy else 0,  # override generation_config's top_k=20 -- see module docstring
        pad_token_id=pad,
    )
    completion_ids = out[:, prompt_len:]
    eos = model.generation_config.eos_token_id
    mask = completion_mask_from_ids(completion_ids, eos if isinstance(eos, list) else [eos])
    lengths = mask.sum(dim=1).long().tolist()
    flat = [ids[:k] for ids, k in zip(completion_ids.tolist(), lengths)]
    return [flat[i * n:(i + 1) * n] for i in range(len(prompt_ids))]


# ---------------------------------------------------------------------------
# Rollout: colocated vLLM
# ---------------------------------------------------------------------------


def check_vllm_quant(quant: str) -> None:
    """Fail early with a plain message if the requested engine quantization isn't available in the installed vLLM
    (0.28 dropped bitsandbytes). Called before LLM() so the error is one line, not a pydantic traceback."""
    if quant != "bnb":
        return
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    except ImportError:  # module layout changed; let vLLM validate it
        return
    if "bitsandbytes" not in QUANTIZATION_METHODS:
        import vllm
        raise SystemExit(f"vllm {vllm.__version__} has no bitsandbytes loader; set cfg.rollout.vllm_quant = \"bf16\" "
                         "in config.py (or --quant bf16 for eval)")


class VLLMRollout:
    """Colocated vLLM engine serving MODEL_ID + the current LoRA adapter. Start it BEFORE the training
    model is loaded: it profiles free memory at startup and claims gpu_memory_utilization of the card."""

    def __init__(self, gpu_memory_utilization: float, sleep_mode: bool, adapter_sync_dir: str, sleep_level: int = 1):
        # The one environment variable this project sets: vLLM reads it in its engine subprocess. Its FlashInfer
        # sampler JIT-compiles with nvcc and fails on images without CUDA headers (see eval/common.py); greedy /
        # plain-softmax sampling gains nothing from it anyway. Must be set before the subprocess is spawned.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        check_vllm_quant(cfg.rollout.vllm_quant)
        self._SamplingParams, self._LoRARequest = SamplingParams, LoRARequest
        self.sleep_mode = sleep_mode
        self.sleep_level = sleep_level
        self.asleep = False
        self.adapter_sync_dir = adapter_sync_dir
        kwargs = dict(
            model=cfg.model.model_id, dtype="bfloat16", max_model_len=cfg.batch.vllm_max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True, max_lora_rank=cfg.lora.r, max_loras=1,
            enable_sleep_mode=sleep_mode, seed=cfg.optim.seed,
        )
        if cfg.rollout.vllm_quant == "bnb":
            kwargs["quantization"] = "bitsandbytes"
        print(f"starting colocated vLLM: quant={cfg.rollout.vllm_quant} max_model_len={cfg.batch.vllm_max_model_len} "
              f"gpu_memory_utilization={gpu_memory_utilization} sleep_mode={sleep_mode}")
        self.llm = LLM(**kwargs)
        self.lora_request = None
        self._prev_adapter_dir = None

    def sleep(self) -> None:
        """Free vLLM's GPU memory for the training phase. No-op without sleep mode.
        Level 1 offloads the weights to CPU RAM and drops the KV cache; level 2 discards the weights as well and
        is only correct together with the reload in wake()."""
        if self.sleep_mode and not self.asleep:
            self.llm.sleep(level=self.sleep_level)
            self.asleep = True

    def wake(self) -> None:
        """Reclaim memory for the rollout. The training side's freed tensors sit in PyTorch's caching
        allocator until empty_cache() returns them to the driver -- without that, wake_up() OOMs.
        After a level-2 sleep the weight buffers are uninitialised memory until reloaded: sampling from them
        looks like a working engine that emits noise to the cap (2026-09-18/19: 53 dead steps)."""
        if self.sleep_mode and self.asleep:
            gc.collect()
            torch.cuda.empty_cache()
            self.llm.wake_up()
            if self.sleep_level == 2:
                try:
                    self.llm.collective_rpc("reload_weights")
                except Exception as e:  # noqa: BLE001 -- surface a clear message instead of silent garbage
                    raise SystemExit(f"vLLM weight reload after level-2 sleep failed ({e!r}); "
                                     "set vllm_sleep_level=1 in the GPU profile") from e
            self.asleep = False

    def sync(self, model, step: int) -> None:
        """Publish the live policy to the engine: save the LoRA adapter, point the next generate() at it.
        vLLM reads the directory lazily on first use, so the previous step's directory (already loaded)
        is only deleted after the new one is written. Each version gets a fresh id -- vLLM caches by id."""
        adapter_dir = f"{self.adapter_sync_dir}/step_{step}"
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)  # PeftModel: adapter weights + config only, ~70MB
        self.lora_request = self._LoRARequest(f"policy_step_{step}", step + 1, adapter_dir)
        if self._prev_adapter_dir and os.path.isdir(self._prev_adapter_dir):
            shutil.rmtree(self._prev_adapter_dir)
        self._prev_adapter_dir = adapter_dir

    def generate(self, prompt_ids: list[list[int]], num_generations: int,
                 max_tokens: int | None = None, greedy: bool = False) -> list[list[list[int]]]:
        """prompt_ids -> per prompt, num_generations completions as token-id lists (EOS included when
        the model stopped; exactly max_tokens when it hit the cap). Sampling is the
        plain softmax, same as rollout() -- vLLM would otherwise pull top_k=20/top_p=0.8 from the
        model's generation_config.json, and the importance ratio would compare two different policies."""
        params = self._SamplingParams(
            n=num_generations, max_tokens=max_tokens or cfg.batch.max_completion_length,
            temperature=0.0 if greedy else cfg.batch.temperature, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids], params,
            lora_request=self.lora_request, use_tqdm=False,
        )
        return [[list(o.token_ids) for o in out.outputs] for out in outputs]


@torch.no_grad()
def assemble_batch(tokenizer, prompt_rows: list[list[int]], completions: list[list[int]], device):
    """Token lists -> the tensors the training side consumes: full_ids/attention_mask
    [N, prompt_len + completion_len], completion_mask [N, completion_len], completion texts, completion_len.
    prompt_rows[i] is the prompt of completions[i] (already repeated per generation)."""
    pad = tokenizer.pad_token_id
    n = len(completions)
    prompt_len = max(len(p) for p in prompt_rows)
    completion_len = max(1, max(len(c) for c in completions))
    full_ids = torch.full((n, prompt_len + completion_len), pad, dtype=torch.long)
    full_mask = torch.zeros((n, prompt_len + completion_len), dtype=torch.long)
    completion_mask = torch.zeros((n, completion_len), dtype=torch.bfloat16)
    for i, (p, c) in enumerate(zip(prompt_rows, completions)):
        full_ids[i, prompt_len - len(p):prompt_len] = torch.tensor(p)          # left-pad the prompt
        full_mask[i, prompt_len - len(p):prompt_len] = 1
        full_ids[i, prompt_len:prompt_len + len(c)] = torch.tensor(c)           # right-pad the completion
        full_mask[i, prompt_len:prompt_len + len(c)] = 1
        completion_mask[i, :len(c)] = 1  # every generated token counts, incl. its EOS; pad (== an EOS id) doesn't
    completions_text = tokenizer.batch_decode(completions, skip_special_tokens=True)
    return full_ids.to(device), full_mask.to(device), completion_mask.to(device), completions_text, completion_len


def pack_micro_batches(prompt_rows: list[list[int]], completions: list[list[int]], token_budget: int,
                       max_sequences: int | None = None) -> list[list[int]]:
    """Indices grouped into micro-batches by a padded-token budget. Sequences are sorted longest-first, so each
    micro-batch pads only to ITS longest completion: a batch with one 4096-token rollout no longer pads all 64
    sequences to 4096, and long retry sequences (retry_cap) simply travel in smaller chunks. Returns the index
    lists; the caller assembles each chunk separately. A chunk always holds at least one sequence and at most
    `max_sequences` (some per-sequence costs -- attention buffers, the sampled-token gather -- don't shrink
    with length the way the token budget assumes)."""
    order = sorted(range(len(completions)), key=lambda i: -(len(prompt_rows[i]) + len(completions[i])))
    chunks, cur, cur_len = [], [], 0
    for i in order:
        seq_len = len(prompt_rows[i]) + len(completions[i])
        width = max(cur_len, seq_len)
        if cur and (width * (len(cur) + 1) > token_budget or (max_sequences and len(cur) >= max_sequences)):
            chunks.append(cur)
            cur, cur_len = [], 0
            width = seq_len
        cur.append(i)
        cur_len = width
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Adaptive group collection: retry all-zero prompts, park hopeless / solved ones
# ---------------------------------------------------------------------------

def score_group(tokenizer, completions: list[list[int]], example: dict) -> list[float]:
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    n = len(completions)
    return total_reward(texts, [example["ground_truth"]] * n,
                        gold_tokens=[example.get("gold_tokens", 0)] * n, completion_ids=completions)


def collect_groups(sample_fn, tokenizer, prompt_ids: list[list[int]], examples: list[dict], num_generations: int,
                   max_rollouts: int, retry_max_tokens: int | None = None) -> tuple[list[list[list[int]]], list[list[float]], dict]:
    """Per prompt: num_generations completions and their rewards, chosen so that a prompt the policy CAN solve
    contributes a group with signal.

    Round 1 samples G per prompt. Prompts whose G rollouts all scored 0 are re-sampled together, G at a time
    (one sampler call per round, so vLLM batches them), until each has a success or has consumed
    max_rollouts. A rescued prompt's group is its successes (up to G-1) plus a random fill of its failures --
    fixed size G, so the training loop is unchanged. Bias to be aware of: the group mean then over-states p for
    very hard prompts (1 success in 24 is presented as 1 in 8); direction is right, magnitude is generous.
    Retry rounds sample with `retry_max_tokens` (cfg.batch.retry_cap): about half of first-round failures are
    truncations, so the retries get more room to finish.
    `sample_fn(prompt_ids, n, max_tokens) -> per prompt n token-id lists` is VLLMRollout.generate or hf_sample."""
    G = num_generations
    groups = sample_fn(prompt_ids, G)
    rewards = [score_group(tokenizer, g, ex) for g, ex in zip(groups, examples)]
    used = [G] * len(prompt_ids)
    stats = {"retried_prompts": 0, "rescued_prompts": 0, "exhausted_prompts": 0, "extra_rollouts": 0, "retry_rounds": 0}
    needs_retry = [i for i, r in enumerate(rewards) if max(r) <= 0]
    stats["retried_prompts"] = len(needs_retry) if max_rollouts > G else 0
    while needs_retry and max_rollouts > G:
        todo = [i for i in needs_retry if used[i] < max_rollouts]
        if not todo:
            break
        n_extra = min(G, max_rollouts - min(used[i] for i in todo))
        extra = sample_fn([prompt_ids[i] for i in todo], n_extra, retry_max_tokens)
        stats["retry_rounds"] += 1
        for i, comps in zip(todo, extra):
            groups[i] += comps
            rewards[i] += score_group(tokenizer, comps, examples[i])
            used[i] += len(comps)
            stats["extra_rollouts"] += len(comps)
        needs_retry = [i for i in todo if max(rewards[i]) <= 0]
    # fixed-size groups
    for i in range(len(prompt_ids)):
        if len(groups[i]) == G:
            continue
        if max(rewards[i]) > 0:
            stats["rescued_prompts"] += 1
        else:
            stats["exhausted_prompts"] += 1
        pos = [j for j, r in enumerate(rewards[i]) if r > 0][:G - 1]
        neg = [j for j, r in enumerate(rewards[i]) if r <= 0]
        fill = torch.randperm(len(neg))[:G - len(pos)].tolist()  # torch RNG: restored on resume
        keep = pos + [neg[k] for k in fill]
        keep = [keep[k] for k in torch.randperm(len(keep)).tolist()]
        groups[i] = [groups[i][j] for j in keep]
        rewards[i] = [rewards[i][j] for j in keep]
    return groups, rewards, stats


def clip_negatives(groups: list[list[list[int]]], rewards: list[list[float]]) -> tuple[list[list[list[int]]], int]:
    """For each group (= ONE prompt's G rollouts) with at least one correct rollout, cut every failed rollout to
    the length of the longest correct one in that same group. Never across prompts: an easy problem solved in 600
    tokens says nothing about where a hard problem's failure went wrong. Token-level: log p(o_t | prefix) for
    t <= L does not depend on the dropped tail, so this equals masking the tail out of the loss. Returns the
    clipped groups and the number of tokens dropped."""
    dropped = 0
    out = []
    for comps, rs in zip(groups, rewards):
        pos_lens = [len(c) for c, r in zip(comps, rs) if r > 0]
        if not pos_lens:
            out.append(comps)
            continue
        limit = max(pos_lens)
        clipped = []
        for c, r in zip(comps, rs):
            if r <= 0 and len(c) > limit:
                dropped += len(c) - limit
                clipped.append(c[:limit])
            else:
                clipped.append(c)
        out.append(clipped)
    return out, dropped


def run_eval(sample_fn, tokenizer, holdout, step: int, output_dir: str) -> dict:
    """Greedy pass rate of the CURRENT policy on a fixed random sample of cfg.eval.sample_size holdout prompts
    (data.dataset.eval_subset with cfg.eval.seed -- the same rows every eval, and the same rows the eval scripts
    score with --sample-size/--seed), on correctness only (reward_fn.math_reward -- the length term is a training
    signal, not the metric). The sampler must already be synced with the live adapter (engine awake + sync, or the
    HF model itself). Appends to OUTPUT_DIR/eval_log.jsonl and returns the record."""
    from reward_fn import math_reward
    t0 = time.time()
    rows = eval_subset(holdout, cfg.eval.sample_size, cfg.eval.seed)
    prompt_ids = tokenize_prompts(tokenizer, rows["prompt"])
    comps = [g[0] for g in sample_fn(prompt_ids, 1, cfg.batch.max_completion_length, True)]
    texts = tokenizer.batch_decode(comps, skip_special_tokens=True)
    rewards = math_reward(texts, rows["ground_truth"])
    lengths = [len(c) for c in comps]
    record = {
        "step": step, "n": len(rows), "seed": cfg.eval.seed, "pass_rate": sum(rewards) / len(rows),
        "mean_tokens": sum(lengths) / len(lengths),
        "frac_at_cap": sum(n >= cfg.batch.max_completion_length for n in lengths) / len(lengths),
        "eval_s": time.time() - t0,
        # per-problem outcomes, so consecutive evals can be diffed: flips in both directions on borderline
        # problems are greedy-decoding noise, one-directional flips are a real change
        "results": {pid: [int(r > 0), n] for pid, r, n in zip(rows["problem_id"], rewards, lengths)},
    }
    flips = ""
    prev = last_eval_results(output_dir)
    if prev:
        gained = sum(1 for pid, (c, _) in record["results"].items() if c and prev.get(pid, [0])[0] == 0)
        lost = sum(1 for pid, (c, _) in record["results"].items() if not c and prev.get(pid, [0])[0] == 1)
        flips = f" | vs previous eval: +{gained} solved, -{lost} lost"
    with open(f"{output_dir}/eval_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"eval @ step {step}: pass rate {record['pass_rate']:.3f} on {record['n']} holdout prompts | "
          f"mean tokens {record['mean_tokens']:.0f} | at cap {record['frac_at_cap']:.2f} | {record['eval_s']:.0f}s{flips}")
    return record


def last_eval_results(output_dir: str) -> dict:
    """Per-problem outcomes of the most recent eval in eval_log.jsonl ({} if none)."""
    path = f"{output_dir}/eval_log.jsonl"
    if not os.path.isfile(path):
        return {}
    last = None
    for line in open(path):
        if line.strip():
            last = json.loads(line)
    return (last or {}).get("results", {})


def evaluated_steps(output_dir: str) -> set:
    """Steps already present in eval_log.jsonl, so a resumed run does not repeat an eval."""
    path = f"{output_dir}/eval_log.jsonl"
    if not os.path.isfile(path):
        return set()
    return {json.loads(line)["step"] for line in open(path) if line.strip()}


def draw_prompts(num_examples: int, k: int, parked: dict, step: int) -> list[int]:
    """k distinct dataset indices, skipping prompts parked until a later step. Uses torch's RNG so a resumed
    run draws the same prompts an uninterrupted one would."""
    chosen: list[int] = []
    for _ in range(50 * k):
        i = int(torch.randint(0, num_examples, (1,)))
        if i in chosen or parked.get(i, -1) > step:
            continue
        chosen.append(i)
        if len(chosen) == k:
            return chosen
    raise SystemExit(f"could not draw {k} unparked prompts ({len(parked)} parked of {num_examples})")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir: str, step: int, model, optimizer, scheduler, sampling_state: dict | None = None) -> str:
    """OUTPUT_DIR/step_{step}/: adapter (save_pretrained) + trainer_state.pt (optimizer, scheduler, RNG,
    step, config snapshot). `step` is the number of completed optimizer steps. Prunes to KEEP_LAST_CHECKPOINTS."""
    ckpt_dir = f"{output_dir}/step_{step}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "config": cfg.snapshot(),
        "sampling": sampling_state or {},  # parked prompts + all-correct streaks (adaptive sampling)
    }
    torch.save(state, f"{ckpt_dir}/trainer_state.pt")
    with open(f"{output_dir}/latest", "w") as f:
        f.write(f"step_{step}\n")
    steps = sorted(int(d[5:]) for d in os.listdir(output_dir) if d.startswith("step_") and d[5:].isdigit())
    for old in steps[:-cfg.checkpoint.keep_last]:
        shutil.rmtree(f"{output_dir}/step_{old}", ignore_errors=True)
    return ckpt_dir


def find_resume_dir(output_dir: str) -> str | None:
    """cfg.checkpoint.resume: "no" -> None; a path -> that path; "auto" -> OUTPUT_DIR/latest if present."""
    if cfg.checkpoint.resume in ("no", "0", "false", "False"):
        return None
    if cfg.checkpoint.resume != "auto":
        if not os.path.isfile(f"{cfg.checkpoint.resume}/trainer_state.pt"):
            raise SystemExit(f"checkpoint.resume={cfg.checkpoint.resume!r} has no trainer_state.pt")
        return cfg.checkpoint.resume
    latest = f"{output_dir}/latest"
    if os.path.isfile(latest):
        name = open(latest).read().strip()
        if os.path.isfile(f"{output_dir}/{name}/trainer_state.pt"):
            return f"{output_dir}/{name}"
    return None


def restore_training_state(ckpt_dir: str, optimizer, scheduler, sampling_state: dict) -> int:
    """Load optimizer/scheduler/RNG (and the adaptive-sampling bookkeeping, into `sampling_state`) from a
    checkpoint; returns the number of completed steps. (The adapter weights were already loaded by attach_lora.)"""
    state = torch.load(f"{ckpt_dir}/trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # The optimizer state carries the LR the *old* schedule had set when the checkpoint was written. If
    # MAX_STEPS changed since (e.g. resuming a 500-step run to go to 800), that value is wrong for the new
    # schedule, and the first resumed step would train at it before scheduler.step() corrects it. Recompute
    # from the current schedule at the restored position instead.
    for group, base_lr, lam in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base_lr * lam(scheduler.last_epoch)
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    for path in ("optim.max_steps", "optim.learning_rate", "batch.prompts_per_step", "batch.num_generations",
                 "batch.max_completion_length", "batch.retry_max_completion_length", "batch.temperature",
                 "batch.max_rollouts_per_prompt", "batch.clip_negatives_to_positive", "optim.schedule",
                 "lora.r", "reward.use_length", "reward.weight"):
        section, key = path.split(".")
        saved, current = state["config"].get(section, {}).get(key), getattr(getattr(cfg, section), key)
        if saved != current:
            print(f"WARNING: checkpoint was written with {path}={saved!r}, current config has {current!r}")
    for k, v in state.get("sampling", {}).items():
        sampling_state[k] = v
    return state["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    # stdout is block-buffered when piped (e.g. `| tee`), while vLLM logs to stderr unbuffered -- without this the
    # step lines would sit in the buffer for hours while the engine's logs stream past.
    if hasattr(sys.stdout, "reconfigure"):  # real files/pipes/ttys; not io.StringIO under test redirects
        sys.stdout.reconfigure(line_buffering=True)
    output_dir = cfg.checkpoint.output_dir_pytorch
    os.makedirs(output_dir, exist_ok=True)
    profile = cfg.resolve_gpu_profile()
    micro_batch = profile.micro_batch_sequences
    print(f"config:\n{cfg.summary()}\n  gpu_profile              = {profile}")

    resume_dir = find_resume_dir(output_dir)
    engine = None
    if cfg.rollout.backend == "vllm":  # before the training model: claims its share of the card first
        engine = VLLMRollout(profile.vllm_gpu_memory_utilization, profile.vllm_sleep_mode,
                             adapter_sync_dir=f"{output_dir}/vllm_adapter", sleep_level=profile.vllm_sleep_level)
    tokenizer = load_tokenizer()
    model = attach_lora(load_base_model(), resume_dir)
    train_dataset, holdout = load_grpo_datasets(cfg.data_dir)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.optim.learning_rate, weight_decay=cfg.optim.weight_decay)
    scheduler = make_lr_scheduler(optimizer, cfg.optim.warmup_steps(), cfg.optim.max_steps)

    torch.manual_seed(cfg.optim.seed)
    start_step = 0
    # adaptive-sampling bookkeeping, checkpointed with the run: parked[idx] = step until which the prompt is
    # skipped; all_correct[idx] = consecutive zero-variance all-correct groups
    sampling_state: dict = {"parked": {}, "all_correct": {}}
    if resume_dir is not None:
        start_step = restore_training_state(resume_dir, optimizer, scheduler, sampling_state)
        print(f"resumed from {resume_dir}: {start_step} steps done, continuing to {cfg.optim.max_steps}"
              f" ({len(sampling_state['parked'])} prompts parked)")
    parked, all_correct = sampling_state["parked"], sampling_state["all_correct"]
    log_path = f"{output_dir}/train_log.jsonl"
    dead_steps = 0  # consecutive steps with no gradient (every group zero-variance)
    G = cfg.batch.num_generations

    for step in range(start_step, cfg.optim.max_steps):
        t_step = time.time()
        batch_idx = draw_prompts(len(train_dataset), cfg.batch.prompts_per_step, parked, step)
        examples = [train_dataset[i] for i in batch_idx]
        prompt_ids = tokenize_prompts(tokenizer, [ex["prompt"] for ex in examples])

        # ---- rollout: G completions per prompt from the live policy (no gradients), retrying all-zero prompts ----
        model.eval()
        if engine is not None:
            engine.wake()
            engine.sync(model, step)  # the rollout policy is the live weights
            sample_fn = engine.generate
        else:
            sample_fn = lambda ids, n, max_tokens=None, greedy=False: hf_sample(model, tokenizer, ids, n, max_tokens, greedy)  # noqa: E731

        # ---- in-loop eval of the policy as it stands after `step` completed steps (engine is awake + synced) ----
        due = cfg.eval.every and ((step == 0 and cfg.eval.at_start) or (step > 0 and step % cfg.eval.every == 0))
        if due and step not in evaluated_steps(output_dir):
            run_eval(sample_fn, tokenizer, holdout, step, output_dir)
            t_step = time.time()  # eval time is reported on its own, not as rollout time

        groups, group_rewards, sstats = collect_groups(
            sample_fn, tokenizer, prompt_ids, examples, G, cfg.batch.max_rollouts_per_prompt,
            retry_max_tokens=cfg.batch.retry_cap,
        )
        if engine is not None:
            engine.sleep()  # 24GB profile: hand the card to the training phase
        t_rollout = time.time() - t_step

        # ---- park bookkeeping ----
        for i, idx, r in zip(range(len(batch_idx)), batch_idx, group_rewards):
            if max(r) <= 0:  # nothing correct even after the retry budget: only a changed policy can fix this
                parked[idx] = step + cfg.batch.park_steps
                all_correct.pop(idx, None)
            elif min(r) == max(r):  # zero-variance all-correct: nothing to learn (length penalty didn't separate)
                all_correct[idx] = all_correct.get(idx, 0) + 1
                if all_correct[idx] >= cfg.batch.park_after_all_correct:
                    parked[idx] = step + cfg.batch.park_steps
                    all_correct.pop(idx, None)
            else:
                all_correct.pop(idx, None)
        for idx in [k for k, until in parked.items() if until <= step]:
            parked.pop(idx)

        # ---- sequences in repeat_interleave order (prompt 0's G, then prompt 1's, ...) ----
        # Rollout metrics (mean tokens, at-cap) are taken BEFORE clipping: they describe what the policy did.
        raw_lengths = torch.tensor([len(c) for g in groups for c in g], dtype=torch.float32)
        frac_at_cap = (raw_lengths >= cfg.batch.max_completion_length).float().mean().item()
        sample_text = tokenizer.decode(groups[0][0], skip_special_tokens=True)
        dropped_tokens = 0
        if cfg.batch.clip_negatives_to_positive:
            groups, dropped_tokens = clip_negatives(groups, group_rewards)
        prompt_rows = [ids for ids in prompt_ids for _ in range(G)]
        completions = [c for g in groups for c in g]
        lengths = torch.tensor([len(c) for c in completions], dtype=torch.float32)  # what the training side sees
        num_sequences = len(completions)  # PROMPTS_PER_STEP * NUM_GENERATIONS
        total_tokens = lengths.sum().clamp(min=1).to(model.device)
        # ---- reward -> group-relative advantages ----
        # Rewards were scored per group in collect_groups (reward_fn.active_rewards; the length term is switched
        # by cfg.reward.use_length; completion_ids there are the generated tokens up to and including EOS).
        rewards = torch.tensor(group_rewards, dtype=torch.float32)  # [PROMPTS_PER_STEP, G]
        advantages = grpo_advantages(rewards).view(-1).to(model.device)
        zero_var_frac = (rewards.std(dim=-1, unbiased=False) == 0).float().mean().item()

        # Sequences with advantage 0 (every rollout of a zero-variance group) contribute exactly 0 to the loss and
        # its gradient. Leave them out of the forward/backward passes; the normalizer stays the FULL token count,
        # so the gradient is bit-identical to running them. This also keeps a parked prompt's 6144-token retry
        # failures out of the micro-batches (they were the 13GB+ peaks in the 2026-09-20 smoke run).
        active = [i for i in range(num_sequences) if advantages[i].item() != 0.0]
        # Micro-batches packed by token budget (micro_batch_sequences x cap): short sequences pack densely, a lone
        # long one doesn't pad the rest. Budget excludes the prompt headroom on purpose -- dense packing at
        # cap+headroom pushed the 24GB peak to 15.6GB against 11.8GB with fixed micro-batches.
        chunks = pack_micro_batches([prompt_rows[i] for i in active], [completions[i] for i in active],
                                    micro_batch * cfg.batch.max_completion_length, max_sequences=2 * micro_batch)
        chunks = [[active[j] for j in c] for c in chunks]  # back to batch indices

        print(f"step {step + 1:4d}/{cfg.optim.max_steps} | rollout done in {t_rollout:.0f}s: mean reward {rewards.mean():.3f}, "
              f"zero-var groups {zero_var_frac:.2f}, mean tokens {raw_lengths.mean():.0f}, at cap {frac_at_cap:.2f}, "
              f"clipped {dropped_tokens} tokens off failures "
              f"| retried {sstats['retried_prompts']} rescued {sstats['rescued_prompts']} parked "
              f"{sstats['exhausted_prompts']} (+{sstats['extra_rollouts']} rollouts) | parked total {len(parked)} "
              f"| training on {len(active)}/{num_sequences} sequences in {len(chunks)} micro-batches ...")

        lr_used = optimizer.param_groups[0]["lr"]  # logged as this step's LR (scheduler advances after the step)
        step_loss = 0.0
        skipped = len(active) == 0
        if skipped:
            # Every group is zero-variance (all rollouts of each prompt scored the same, e.g. all 0): every advantage
            # is 0, so the loss and its gradient are identically 0. The two log-prob passes over 64 sequences would
            # cost ~10 min on the A10G to apply nothing -- skip them. The step still counts (schedule, log, checkpoint).
            print(f"step {step + 1:4d}/{cfg.optim.max_steps} | no advantage in any group -> skipping log-prob + backward passes")
        else:
            # Each micro-batch is assembled on its own (padded to its own longest sequence) and kept for the passes
            # below: full_ids / attention mask / completion mask / completion_len / this chunk's advantages.
            batches = []
            for idx in chunks:
                full_ids, full_mask, completion_mask, _, completion_len = assemble_batch(
                    tokenizer, [prompt_rows[i] for i in idx], [completions[i] for i in idx], model.device,
                )
                batches.append((full_ids, full_mask, completion_mask, completion_len, advantages[idx]))

            # ---- pi_old: log-probs of the sampled tokens under the weights that generated them ----
            with torch.no_grad():
                logp_old = [
                    sequence_logprobs(model, b[0], b[1], b[3])
                    for b in tqdm(batches, desc="  logp_old", unit="mb", leave=False)
                ]

            # ---- mu gradient steps on the same rollout; gradients accumulate over all micro-batches ----
            model.train()
            for _ in range(cfg.optim.num_iterations):
                optimizer.zero_grad()
                step_loss = 0.0
                for (full_ids, full_mask, completion_mask, completion_len, adv), old in zip(
                        tqdm(batches, desc="  backward", unit="mb", leave=False), logp_old):
                    # pi_theta: same computation, live weights, gradients on
                    logp = sequence_logprobs(model, full_ids, full_mask, completion_len)
                    loss = grpo_loss(logp, old, adv, completion_mask, normalizer=total_tokens)
                    loss.backward()
                    step_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip_norm)
                optimizer.step()
        scheduler.step()

        record = {
            "step": step + 1, "loss": step_loss, "mean_reward": rewards.mean().item(),
            # one completion's head, so broken rollouts (noise, wrong template, empty) are visible in the log
            "sample_completion": sample_text[:300],
            "skipped_backprop": skipped,
            **sstats, "parked_total": len(parked),
            "zero_variance_groups": zero_var_frac, "mean_completion_tokens": raw_lengths.mean().item(),
            "frac_at_cap": frac_at_cap, "micro_batches": len(chunks),
            "train_tokens": int(lengths[active].sum().item()) if active else 0, "clipped_tokens": dropped_tokens,
            "active_sequences": len(active),
            "lr": lr_used, "rollout_s": t_rollout, "step_s": time.time() - t_step,
        }
        if torch.cuda.is_available():
            record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
        print(f"step {step + 1:4d}/{cfg.optim.max_steps} | loss {step_loss:.4f} | mean reward {record['mean_reward']:.3f} "
              f"| zero-var groups {zero_var_frac:.2f} | mean tokens {record['mean_completion_tokens']:.0f} "
              f"| at cap {record['frac_at_cap']:.2f} | rollout {t_rollout:.0f}s | step {record['step_s']:.0f}s"
              + (f" | peak {record['peak_mem_gb']:.1f}GB" if "peak_mem_gb" in record else ""))
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # dead = no gradient AND nothing correct. All-correct groups are also zero-variance but mean the model is
        # fine (and the length penalty still separates them); broken rollouts show up as reward 0 everywhere.
        dead_steps = dead_steps + 1 if (zero_var_frac == 1.0 and rewards.max().item() <= 0.0) else 0
        if dead_steps >= cfg.optim.max_dead_steps:
            print(f"first completion of this step:\n{sample_text[:600]!r}")
            raise SystemExit(
                f"{dead_steps} consecutive steps with reward 0 in every rollout (no gradient). Likely causes: "
                "rollouts are broken (check the sample above and 'sample_completion' in train_log.jsonl), the cap is "
                "too small (frac_at_cap ~1.0), or the batch is all-unsolvable. Fix and rerun; resume picks up from "
                "the last checkpoint.")

        if (step + 1) % cfg.checkpoint.every == 0 or step + 1 == cfg.optim.max_steps:
            print(f"checkpoint -> {save_checkpoint(output_dir, step + 1, model, optimizer, scheduler, sampling_state)}")

    os.makedirs(f"{output_dir}/final", exist_ok=True)
    model.save_pretrained(f"{output_dir}/final")
    if cfg.eval.every and cfg.optim.max_steps not in evaluated_steps(output_dir) and cfg.optim.max_steps > start_step:
        model.eval()
        if engine is not None:
            engine.wake()
            engine.sync(model, cfg.optim.max_steps)
            sample_fn = engine.generate
        else:
            sample_fn = lambda ids, n, max_tokens=None, greedy=False: hf_sample(model, tokenizer, ids, n, max_tokens, greedy)  # noqa: E731
        run_eval(sample_fn, tokenizer, holdout, cfg.optim.max_steps, output_dir)
        if engine is not None:
            engine.sleep()
    print(f"done: final adapter at {output_dir}/final")


if __name__ == "__main__":
    train()
