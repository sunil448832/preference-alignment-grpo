"""Every training knob in one place. train_grpo_pytorch.py and train_grpo_rlvr.py read from here, as do
reward_fn.py (reward shape) and eval/common.py (model id, LoRA rank). Edit the defaults in the section
dataclasses below; nothing is read from the environment. The scripts use the module-level instance `cfg`,
e.g. cfg.optim.learning_rate, cfg.batch.max_completion_length, cfg.checkpoint.resume.

GPU profile: the two scripts run on a 24GB card (A10G), a 40GB card (A100) or a 48GB card (L40S) from the same code. The
profile picks the micro-batch and how vLLM shares the card; everything else -- full batch, cap, LR -- is
the same, so a run on either card is the same experiment at a different speed.

Smoke run: smoke=True shrinks steps/batch/cap to a few minutes' worth and writes to a separate output
directory. It still exercises every code path (vLLM, checkpointing, resume).
"""
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    bnb_4bit: dict = field(default_factory=lambda: dict(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))  # compute dtype bf16 in scripts
    bf16: bool = True


@dataclass
class AdapterConfig:
    """LoRA on top of the NF4 base."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0            # RL: the ratio must see weight movement only, not dropout noise
    target_modules: str = "all-linear"


@dataclass
class BatchConfig:
    """Rollout shape. The full batch is the same on every card; only the micro-batch (GPUProfile) differs."""
    num_generations: int = 8        # G: completions sampled per prompt (the GRPO group)
    prompts_per_step: int = 16       # unique prompts per optimizer step -> 64 sequences per step
    max_completion_length: int = 4096  # eval/BASELINE_NOTES.md Sec 1: 4096 settled; 8192 rescues a trickle
    prompt_headroom: int = 1024     # longest pool prompt is ~630 tokens; vLLM max_model_len = cap + this
    temperature: float = 0.7        # sampling temperature; the scorer divides logits by the same T, so the
                                    # importance ratio compares the same tempered policy (trl does likewise).
                                    # 1.0 rambled: 52-94% of rollouts hit the cap vs 34% greedy (2026-09-18 run);
                                    # 0.7 is the model's shipped default. top_p=1, top_k=0 (no truncation) always.

    # ---- adaptive sampling (train_grpo_pytorch.py) ----
    # A group whose G rollouts all score 0 carries no gradient. Instead of wasting it, keep sampling that prompt in
    # rounds of G until a success appears or the budget runs out, then train on a fixed-size group of the
    # successes (up to G-1) plus a random fill of failures. Prompts that exhaust the budget with no success are
    # parked (redrawn only after park_steps, since only a changed policy can change that outcome); prompts the
    # policy solves G/G `park_after_all_correct` times in a row are parked too (nothing left to learn from them
    # unless the length penalty separates them, which is why that check only applies to zero-variance groups).
    max_rollouts_per_prompt: int = 16   # total samples a prompt may consume in one step; == num_generations disables
    retry_max_completion_length: int = 6144  # cap for the RETRY rounds only (0 = same as max_completion_length).
                                    # ~half of failures are truncations, so retries get more room. Costs: vLLM
                                    # max_model_len is sized to this (fewer sequences in flight), and the training
                                    # micro-batches holding these long sequences shrink to fit the token budget.
    park_steps: int = 100               # how many steps a parked prompt stays out of the draw
    park_after_all_correct: int = 2     # consecutive zero-variance all-correct groups before parking
    clip_negatives_to_positive: bool = True  # in a group that has a correct rollout, truncate the FAILED rollouts to
                                    # the longest correct one's length (rewards are scored on the full text first).
                                    # Scope is ONE PROMPT's G rollouts: a hard problem's failures are only cut to that
                                    # problem's own correct length, never to another problem's shorter answer.
                                    # A failure's tail beyond where a correct answer finished is where the model is
                                    # lost -- the negative push there is mostly noise (DAPO's overlong argument) --
                                    # and dropping it shrinks the padded micro-batches. Groups with no correct
                                    # rollout are untouched (they carry no gradient anyway).

    @property
    def full_batch_sequences(self) -> int:
        """Sequences per optimizer step, identical on every card."""
        return self.prompts_per_step * self.num_generations

    @property
    def retry_cap(self) -> int:
        return self.retry_max_completion_length or self.max_completion_length

    @property
    def vllm_max_model_len(self) -> int:
        return max(self.max_completion_length, self.retry_cap) + self.prompt_headroom


@dataclass
class OptimConfig:
    learning_rate: float = 1e-5     # LoRA RL: ~10x the full-FT RL rate of 1e-6 (LoRA Without Regret); 2e-5 is the
                                    # upper end before entropy collapse shows up
    weight_decay: float = 0.0       # torch AdamW defaults to 0.01, which decays the adapter toward the base model
                                    # every step; trl's GRPO default is 0
    max_steps: int = 300            # tuning knob: watch reward-vs-step, stop at the plateau
    schedule: str = "wsd"           # "wsd": warmup, then CONSTANT lr, then linear decay over the last decay_frac of
                                    #        max_steps -- every step trains at full rate and max_steps can be extended
                                    #        on resume without reshaping the run
                                    # "linear": warmup then linear decay to 0 at max_steps (HF default)
                                    # "constant": warmup then flat
    decay_frac: float = 0.1         # wsd only: fraction of max_steps spent decaying at the end
    max_dead_steps: int = 5         # abort after this many consecutive steps where every rollout scored 0 (no
                                    # gradient, nothing correct). Catches broken rollouts / a cap far too small.
                                    # All-correct steps do not count. With few prompts per step, unlucky streaks
                                    # are likelier; raise it for smoke runs with 1-2 prompts.
    warmup_ratio: float = 0.03
    num_iterations: int = 1         # mu: gradient steps per rollout batch. 1 => ratio is always 1, clip inactive
    clip_eps: float = 0.2
    grad_clip_norm: float = 1.0
    beta: float = 0.0               # KL weight; 0 = no reference model (docs/PLAN.md Stage 2)
    seed: int = 0

    def warmup_steps(self) -> int:
        return max(1, round(self.warmup_ratio * self.max_steps))


@dataclass
class RewardConfig:
    """reward_fn.py: correctness (1/0) plus an optional one-sided length penalty on correct-but-overlong answers."""
    use_length: bool = False         # False: train on correctness alone (reward_fn.active_rewards)
    free_ratio: float = 3.0         # no penalty up to this multiple of the reference solution's token count
    full_ratio: float = 4.0         # full penalty at this multiple (linear between)
    weight: float = 0.5             # correct answer scores 1 - weight*excess, wrong 0; keep < 1 so correct beats wrong


@dataclass
class RolloutConfig:
    """Where sampled tokens come from (train_grpo_pytorch.py; trl always uses vLLM colocate)."""
    backend: str = "vllm"           # "vllm" | "hf" (plain model.generate: readable, slow, no engine)
    vllm_quant: str = "bf16"        # "bf16": full weights in the engine (7.6GB). "bnb" (in-flight NF4, 2.5GB) only
                                    # works on vLLM versions that still ship the bitsandbytes loader; 0.28 dropped it
                                    # ("Unknown quantization method: bitsandbytes"). The engine then samples from a
                                    # slightly different numeric policy than the NF4 training copy; logp_old is still
                                    # computed by the training model, so the ratio stays well-defined (see script docstring).


@dataclass
class GPUProfile:
    """How one card is split between the training model and the colocated vLLM engine."""
    name: str
    micro_batch_sequences: int          # forward/backward chunk at the nominal cap. The actual micro-batches are packed
                                        # by TOKEN budget = this x (max_completion_length + prompt_headroom): sequences
                                        # are sorted by length, so short ones pack more per chunk and long retry
                                        # sequences fewer. Gradients accumulate to the full batch before each step.
    vllm_gpu_memory_utilization: float  # fraction of the card vLLM claims at startup (weights + KV cache)
    vllm_sleep_mode: bool               # True: vLLM frees its GPU memory during the gradient steps and restores it
                                        # before the next rollout, so the two phases never need their peak at once.
                                        # A few seconds per step; required on 24GB, not on 40GB.
    vllm_sleep_level: int = 1           # 1: weights backed up to CPU RAM (7.6GB bf16) and restored on wake; KV cache
                                        # dropped. 2: weights DISCARDED too -- vLLM then serves uninitialised memory
                                        # after wake unless they are reloaded, which produced 53 steps of garbage
                                        # rollouts (reward 0, 95% at cap) on 2026-09-18/19. Level 2 is only used with an
                                        # explicit weight reload on wake (see VLLMRollout.wake); keep 1 unless CPU RAM
                                        # is too small to hold the weights.


@dataclass
class GPUConfig:
    profile: str = "auto"           # "auto" (from the card's memory) | "24gb" | "40gb" | "48gb"
    profiles: dict = field(default_factory=lambda: {
        # Budgets assume bf16 engine weights (7.6GB). Sleep mode measured at 0.7 s to sleep + 1.1 s to wake per step
        # (A10G, level 1), i.e. free against 4-18 min steps, so both profiles use it and give each phase the card:
        #   24gb: engine 16.8GB awake (~7.5GB KV, ~16 seqs in flight); training peak 11.8GB at an 8K-token micro-batch
        #   40gb: engine 24GB awake (~15GB KV, ~100K tokens, ~20 seqs in flight); training ~20GB at a 16K-token budget.
        #         Both resident would be ~42GB on a 40GB card -- do not turn sleep off here.
        #   48gb (L40S): engine 0.80 = ~38GB awake (~29GB KV, ~200K tokens, ~39 seqs in flight -> 64 rollouts in two
        #         waves instead of three); the idle training model (~3.3GB) + contexts still fit beside it. Training
        #         at a 24K-token micro-batch budget peaks ~30GB with the engine asleep.
        "24gb": GPUProfile("24gb", micro_batch_sequences=2, vllm_gpu_memory_utilization=0.70, vllm_sleep_mode=True),
        "40gb": GPUProfile("40gb", micro_batch_sequences=4, vllm_gpu_memory_utilization=0.60, vllm_sleep_mode=True),
        "48gb": GPUProfile("48gb", micro_batch_sequences=6, vllm_gpu_memory_utilization=0.80, vllm_sleep_mode=True),
    })

    def resolve(self, full_batch_sequences: int, total_memory_gb: float | None = None) -> GPUProfile:
        """Explicit profile wins; "auto" reads the card's memory (or the number given, for tests)."""
        name = self.profile
        if name == "auto":
            if total_memory_gb is None:
                import torch
                if not torch.cuda.is_available():
                    raise SystemExit('no CUDA device; set gpu.profile to "24gb" or "40gb" explicitly')
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            name = "24gb" if total_memory_gb < 32 else "40gb" if total_memory_gb < 44 else "48gb"
        if name not in self.profiles:
            raise SystemExit(f"unknown gpu.profile {name!r}; choose from {sorted(self.profiles)}")
        # micro_batch_sequences is a token-budget multiplier for the packer (train_grpo_pytorch.pack_micro_batches),
        # not a fixed chunk size, so the full batch need not be a multiple of it.
        return self.profiles[name]


@dataclass
class EvalConfig:
    """In-loop held-out eval (train_grpo_pytorch.py): greedy decode of a fixed random sample of the holdout through the
    rollout engine with the current adapter, scored on correctness only (no length term). Logged to
    OUTPUT_DIR/eval_log.jsonl. The training loss is not a progress signal in GRPO (with one update per rollout the
    ratio is 1 and the loss is just the zero-mean advantage sum); this pass rate is."""
    every: int = 25                 # steps between evals; 0 disables. 200 prompts at cap 4K ~ 10 min on an A10G
    sample_size: int = 200          # rows drawn from the 500-row holdout (data.dataset.eval_subset); the SAME rows every
                                    # eval and in the eval scripts for the same seed. ~+/-7 points per point at 200,
                                    # +/-10 at 100; the trend across points is what to read, evaluate.py on all 500
                                    # is the final number
    seed: int = 0                   # seed of the subset draw; independent of optim.seed and of the training RNG
    at_start: bool = True           # eval the untrained adapter at step 0 for a paired baseline


@dataclass
class CheckpointConfig:
    output_dir_pytorch: str = "checkpoints/grpo_pytorch"
    output_dir_rlvr: str = "checkpoints/grpo_rlvr"
    every: int = 25                 # steps between checkpoints (adapter + optimizer + scheduler + RNG + step)
    keep_last: int = 3              # older step_* / checkpoint-* directories are deleted
    resume: str = "auto"            # "auto": newest checkpoint in the output dir if any; "no": start fresh; or a path


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: AdapterConfig = field(default_factory=AdapterConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data_dir: str = "data"
    smoke: bool = True             # a few minutes on any card, every code path. Set True for the first run on a new box

    def __post_init__(self):
        if self.smoke:
            self.optim.max_steps = 5
            self.optim.max_dead_steps = 3

            self.batch.prompts_per_step = 4           # 32 sequences: exercises micro-batch packing at 4 (40gb profile)
            self.batch.num_generations = 8
            self.batch.max_completion_length = 4096   # long enough that some completions finish: a smoke run must
                                                      # show non-zero reward on some step, or rollouts are broken
            self.checkpoint.every = 2
            self.eval.every = 2
            self.eval.sample_size = 20
            self.checkpoint.output_dir_pytorch = "checkpoints/smoke_pytorch"
            self.checkpoint.output_dir_rlvr = "checkpoints/smoke_rlvr"

    def resolve_gpu_profile(self, total_memory_gb: float | None = None) -> GPUProfile:
        return self.gpu.resolve(self.batch.full_batch_sequences, total_memory_gb)

    def snapshot(self) -> dict:
        """Plain nested dict for checkpoint metadata."""
        return asdict(self)

    def summary(self) -> str:
        rows = [
            ("model", self.model.model_id),
            ("batch", f"{self.batch.prompts_per_step} prompts x {self.batch.num_generations} gens = "
                      f"{self.batch.full_batch_sequences} seqs, cap {self.batch.max_completion_length}"),
            ("optim", f"lr {self.optim.learning_rate} {self.optim.schedule} (warmup {self.optim.warmup_steps()}), "
                      f"{self.optim.max_steps} steps, mu {self.optim.num_iterations}, clip {self.optim.clip_eps}, "
                      f"wd {self.optim.weight_decay}"),
            ("lora", f"r {self.lora.r}, alpha {self.lora.alpha}"),
            ("reward", f"length penalty {'on' if self.reward.use_length else 'off'} "
                       f"({self.reward.free_ratio}x-{self.reward.full_ratio}x gold, weight {self.reward.weight})"),
            ("rollout", f"{self.rollout.backend}, vllm quant {self.rollout.vllm_quant}"),
            ("gpu", self.gpu.profile),
            ("checkpoint", f"every {self.checkpoint.every}, keep {self.checkpoint.keep_last}, resume {self.checkpoint.resume}"),
            ("eval", f"every {self.eval.every} steps on {self.eval.sample_size} holdout prompts (seed {self.eval.seed})" if self.eval.every else "off"),
            ("smoke", self.smoke),
        ]
        return "\n".join(f"  {k:12s} {v}" for k, v in rows)


cfg = TrainConfig()
