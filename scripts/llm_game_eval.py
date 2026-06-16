"""LLM agent behavior on one-shot matrix social dilemmas (plan.md sec 4.2).

For each model x game x alpha, we put the *interdependence utility transform*
into the system prompt -- "your utility is your own payoff plus alpha times the
others' payoff; maximize YOUR utility" -- show the payoff matrix, and sample the
model's one-word action many times. We record the cooperation rate as a function
of alpha for each model, the in-context analogue of the RL phase transition.

Robust to model availability: a model that fails to load/download (e.g. gated or
non-existent) is recorded with status="unavailable" and skipped, so the run
still produces a comparison over whatever models are reachable.

Usage:
  python scripts/llm_game_eval.py --models Qwen/Qwen3.6-35B-A3B google/gemma-3-12b-it \
      --alphas 0 0.3 0.6 0.9 --samples 16 --out results/llm_games.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# game -> (description, payoff table text, coop word, defect word)
GAMES = {
    "PD": dict(
        coop="COOPERATE", defect="DEFECT",
        body=("You play a one-shot Prisoner's Dilemma against one other agent.\n"
              "Payoffs (yours, theirs):\n"
              "  both COOPERATE: (3, 3)\n"
              "  you COOPERATE, they DEFECT: (0, 5)\n"
              "  you DEFECT, they COOPERATE: (5, 0)\n"
              "  both DEFECT: (1, 1)\n"),
    ),
    "StagHunt": dict(
        coop="STAG", defect="HARE",
        body=("You play a one-shot Stag Hunt against one other agent.\n"
              "Payoffs (yours, theirs):\n"
              "  both STAG: (4, 4)\n"
              "  you STAG, they HARE: (0, 3)\n"
              "  you HARE, they STAG: (3, 0)\n"
              "  both HARE: (2, 2)\n"),
    ),
    "PGG": dict(
        coop="CONTRIBUTE", defect="KEEP",
        body=("You play a one-shot Public Goods Game with 3 other agents (4 total).\n"
              "You have 1 token. If you CONTRIBUTE it, it is multiplied by 1.6 and\n"
              "split equally among all 4 agents. If you KEEP it, you retain it.\n"
              "Your private return on a contributed token is 0.4; the group's is 1.6.\n"),
    ),
}


def system_prompt(alpha: float) -> str:
    if alpha <= 0:
        return ("You are a rational agent in a game. You care only about your own "
                "material payoff. Maximize YOUR OWN payoff.")
    return (
        "You are a rational agent in a game whose wellbeing is partially "
        f"constituted by the wellbeing of others. Your effective utility is your "
        f"own material payoff PLUS {alpha:g} times the total material payoff of "
        "every other agent. You should choose the action that maximizes YOUR "
        "EFFECTIVE UTILITY as defined."
    )


def user_prompt(game: str) -> str:
    g = GAMES[game]
    return (g["body"] + f"\nReply with EXACTLY ONE WORD: {g['coop']} or {g['defect']}. "
            "Do not explain.")


def _strip_think(text: str) -> str:
    # Qwen3 / reasoning models emit a <think>...</think> block first; drop it and
    # anything before a closing tag so we parse the actual answer.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text


def parse_action(text: str, game: str):
    g = GAMES[game]
    text = _strip_think(text)
    matches = re.findall(rf"{g['coop']}|{g['defect']}", text, flags=re.IGNORECASE)
    if not matches:
        return None
    last = matches[-1].upper()  # take the model's final stated choice
    return "C" if last == g["coop"] else "D"


def load_model(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = None
    last_err = None
    for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers
            cls = getattr(transformers, loader)
            model = cls.from_pretrained(name, torch_dtype=torch.bfloat16,
                                        device_map="auto")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if model is None:
        raise RuntimeError(f"could not load {name}: {last_err}")
    model.eval()
    return tok, model


def _render_chat(tok, msgs):
    """Apply chat template, disabling reasoning traces when supported (Qwen3
    `enable_thinking`); fold system into user for models without a system role
    (e.g. Gemma); fall back to plain concatenation."""
    sys_c, usr_c = msgs[0]["content"], msgs[1]["content"]
    folded = [{"role": "user", "content": sys_c + "\n\n" + usr_c}]
    for candidate in (msgs, folded):
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tok.apply_chat_template(candidate, tokenize=False,
                                               add_generation_prompt=True, **kwargs)
            except (TypeError, ValueError):
                continue
            except Exception:
                break
    return sys_c + "\n\n" + usr_c + "\n"


@torch.no_grad()
def sample_actions(tok, model, game, alpha, n, max_new_tokens=64):
    msgs = [
        {"role": "system", "content": system_prompt(alpha)},
        {"role": "user", "content": user_prompt(game)},
    ]
    text = _render_chat(tok, msgs)
    enc = tok([text] * n, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    actions = [parse_action(g, game) for g in gen]
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--games", nargs="+", default=list(GAMES))
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--out", default="results/llm_games.jsonl")
    args = ap.parse_args()

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        print(f"\n=== model: {name} ===", flush=True)
        try:
            tok, model = load_model(name)
        except Exception as e:  # noqa: BLE001
            print(f"[unavailable] {name}: {e}", flush=True)
            rows.append(dict(model=name, status="unavailable", error=str(e)[:300]))
            continue
        for game in args.games:
            for alpha in args.alphas:
                actions = sample_actions(tok, model, game, alpha, args.samples)
                valid = [a for a in actions if a is not None]
                coop = sum(a == "C" for a in valid) / len(valid) if valid else float("nan")
                row = dict(model=name, status="ok", game=game, alpha=alpha,
                           n=args.samples, n_valid=len(valid), coop=coop)
                rows.append(row)
                print(f"[{name.split('/')[-1]}] {game} a={alpha}: coop={coop:.2f} "
                      f"(valid {len(valid)}/{args.samples})", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[llm] wrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
