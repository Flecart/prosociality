"""Full GRPO + LoRA training of LLM agents on the interdependence-transformed PD.

Unlike the in-context probe (which only *prompts*), this actually *trains* a model
with GRPO so its policy optimizes the structural reward U_i = [(I-A)^{-1} pi]_i.
We train at a caring coupling (alpha>2/3, cooperation is optimal) and at a selfish
control (alpha=0, defection is optimal), and log the cooperation rate every
`eval_every` steps so we can report a full training curve, not a smoke.

Writes results/grpo_<model>_<alpha>.jsonl (one row per eval checkpoint).

Usage:
  python scripts/grpo_train.py --model Qwen/Qwen3-0.6B --alpha 0.8 --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prosocial import pd_payoffs, symmetric_matrix, effective_utilities  # noqa: E402

PROMPT = ("We are playing a one-shot Prisoner's Dilemma. You may either COOPERATE "
          "or DEFECT. Reply with exactly one word: COOPERATE or DEFECT.")
OPP = "C"  # fixed cooperating opponent


def parse_action(text):
    m = re.search(r"COOPERATE|DEFECT", text, flags=re.IGNORECASE)
    return None if not m else ("C" if m.group(0).upper() == "COOPERATE" else "D")


def _txt(c):
    if isinstance(c, list):
        return " ".join(m.get("content", "") for m in c if isinstance(m, dict))
    return str(c)


def make_reward(alpha):
    A = symmetric_matrix(2, alpha) if alpha > 0 else None

    def reward(completions, **kw):
        out = []
        for c in completions:
            a = parse_action(_txt(c))
            if a is None:
                out.append(-1.0); continue
            pi = pd_payoffs(a, OPP)
            u = effective_utilities(A, pi)[0] if A is not None else pi[0]
            out.append(float(u))
        return out
    return reward


@torch.no_grad()
def coop_rate(model, tok, n=32):
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) if _supports_think(tok) \
        else tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text] * n, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=1.0,
                         pad_token_id=tok.pad_token_id)
    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    acts = [parse_action(g) for g in gen]
    valid = [a for a in acts if a]
    return sum(a == "C" for a in valid) / len(valid) if valid else float("nan")


def _supports_think(tok):
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    outp = ROOT / f"results/grpo_{short}_a{args.alpha:g}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"[grpo] model={args.model} alpha={args.alpha} steps={args.steps}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": PROMPT}]} for _ in range(256)])
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    rows = []

    class EvalCB(TrainerCallback):
        def on_step_end(self, cfg, state, control, **kw):
            if state.global_step % args.eval_every == 0 or state.global_step == args.steps:
                model.eval()
                cr = coop_rate(model, tok)
                model.train()
                rows.append(dict(model=args.model, alpha=args.alpha,
                                 step=state.global_step, coop=cr))
                with open(outp, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"[grpo] step={state.global_step} coop={cr:.3f}", flush=True)

    cfg = GRPOConfig(
        output_dir=str(ROOT / f"outputs/grpo_{short}_a{args.alpha:g}"),
        per_device_train_batch_size=args.batch, num_generations=args.num_generations,
        max_completion_length=16, max_steps=args.steps, learning_rate=1e-5,
        beta=0.0, temperature=1.0, logging_steps=10, save_strategy="no",
        bf16=True, use_vllm=False, report_to="none")

    # baseline (step 0) cooperation before any training
    model.eval(); rows.append(dict(model=args.model, alpha=args.alpha, step=0,
                                   coop=coop_rate(model, tok))); model.train()
    print(f"[grpo] step=0 (baseline) coop={rows[0]['coop']:.3f}", flush=True)

    trainer = GRPOTrainer(model=model, reward_funcs=make_reward(args.alpha), args=cfg,
                          train_dataset=dataset, processing_class=tok,
                          peft_config=peft_config, callbacks=[EvalCB()])
    trainer.train()
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[grpo] wrote {len(rows)} rows -> {outp}", flush=True)


if __name__ == "__main__":
    main()
