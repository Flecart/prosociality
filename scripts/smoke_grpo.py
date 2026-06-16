"""Tiny GRPO + LoRA smoke run for the prosociality project.

Validates the training stack (TRL GRPO + PEFT LoRA on Qwen3-0.6B) end to end
on one B200, while exercising the project's core idea: the reward is the
*interdependence-transformed* PD payoff U = (I - A)^{-1} pi (Bergstrom 1999),
not the raw payoff. The policy plays a one-shot Prisoner's Dilemma against a
fixed cooperating opponent and must emit COOPERATE or DEFECT.

With alpha > 2/3 (we use 0.8), COOPERATE has the higher effective utility, so a
working pipeline should nudge the cooperation rate up over a handful of steps.
This is a plumbing test, not a result -- a few steps on a 0.6B model.

Run via SLURM: scripts/slurm/smoke.sbatch  (or srun ... python scripts/smoke_grpo.py)
"""

from __future__ import annotations

import os
import re
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prosocial import pd_payoffs, symmetric_matrix, effective_utilities  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")
ALPHA = float(os.environ.get("SMOKE_ALPHA", "0.8"))   # > 2/3 -> cooperation is optimal
OPP_ACTION = "C"                                       # fixed cooperating opponent
N_PROMPTS = 16
OUT_DIR = os.environ.get("SMOKE_OUT", "outputs/smoke_grpo")

PROMPT = (
    "We are playing a one-shot Prisoner's Dilemma. You may either COOPERATE or "
    "DEFECT. Reply with exactly one word: COOPERATE or DEFECT."
)


def parse_action(text: str) -> str | None:
    """First of COOPERATE/DEFECT to appear in the completion -> 'C'/'D'."""
    m = re.search(r"COOPERATE|DEFECT", text, flags=re.IGNORECASE)
    if not m:
        return None
    return "C" if m.group(0).upper() == "COOPERATE" else "D"


def _completion_text(c) -> str:
    # GRPO completions are conversational: [{"role": "assistant", "content": ...}]
    if isinstance(c, list):
        return " ".join(m.get("content", "") for m in c if isinstance(m, dict))
    return str(c)


def interdependence_reward(completions, **kwargs):
    """Reward = U_i, agent i's effective utility under symmetric coupling alpha.

    Unparseable completions get a small penalty so the format is learnable.
    """
    A = symmetric_matrix(2, ALPHA)
    rewards = []
    for c in completions:
        action = parse_action(_completion_text(c))
        if action is None:
            rewards.append(-1.0)
            continue
        pi = pd_payoffs(action, OPP_ACTION)        # raw (pi_i, pi_j)
        u = effective_utilities(A, pi)             # (I - A)^{-1} pi
        rewards.append(float(u[0]))
    return rewards


def coop_rate(trainer, tokenizer, n: int = 16) -> tuple[float, float]:
    """Quick eval: sample n completions, report (coop_fraction, mean_reward)."""
    msgs = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([text] * n, return_tensors="pt", padding=True).to(trainer.model.device)
    with torch.no_grad():
        out = trainer.model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=1.0)
    gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    completions = [[{"role": "assistant", "content": g}] for g in gen]
    actions = [parse_action(_completion_text(c)) for c in completions]
    coop = sum(a == "C" for a in actions) / n
    mean_r = sum(interdependence_reward(completions)) / n
    return coop, mean_r


def main():
    print(f"[smoke] model={MODEL} alpha={ALPHA} (threshold=2/3) opp={OPP_ACTION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": PROMPT}]} for _ in range(N_PROMPTS)]
    )

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = GRPOConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=8,
        num_generations=4,           # group size; batch must be a multiple of this
        max_completion_length=16,
        max_steps=4,                 # smoke: a handful of updates
        learning_rate=1e-5,
        beta=0.0,                    # no KL/ref model -> lighter & faster for a smoke
        temperature=1.0,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        use_vllm=False,              # HF generation (installed vLLM is out of TRL's range)
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=interdependence_reward,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    before = coop_rate(trainer, tokenizer)
    print(f"[smoke] BEFORE  coop_rate={before[0]:.2f}  mean_reward={before[1]:.3f}")
    trainer.train()
    after = coop_rate(trainer, tokenizer)
    print(f"[smoke] AFTER   coop_rate={after[0]:.2f}  mean_reward={after[1]:.3f}")
    print("[smoke] DONE (plumbing OK; directional signal only, not a result)")


if __name__ == "__main__":
    main()
