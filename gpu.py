#!/usr/bin/env python3
"""Quick GPU availability across the SLURM cluster, totalled and broken down by
GPU type (and optionally by partition), with CPU and memory headroom so you can
see which resource is actually the bottleneck when asking for resources.

Parses `scontrol show nodes` for each node's configured GPUs (Gres=) and in-use
GPUs (AllocTRES), plus CPUs (CPUTot/CPUAlloc) and memory (RealMemory/AllocMem),
and the node State. Resources on down/drained/not-responding nodes are counted as
unavailable, not free.

A node can have free GPUs that are still unschedulable because its CPU or memory
is fully allocated. The "free" CPU/mem columns therefore count only headroom on
nodes that still have at least one free GPU of that type (schedulable headroom),
while "all-free" counts headroom across every available node of that type.

Usage:
  python gpu.py            # by type
  python gpu.py --by-part  # by type x partition
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict

# Gres GPU spec is either typed ('gpu:b200:8(IDX:0-7)') or bare ('gpu:4'); the
# type group is optional. On this cluster Gres is always bare ('gpu:4') and the
# GPU model lives in node features instead, so we fall back to that.
GRES_RE = re.compile(r"gpu:(?:([^:\s(]+):)?(\d+)")    # gpu:[<type>:]<count>[(...)]
ALLOC_RE = re.compile(r"gres/gpu=(\d+)")              # AllocTRES gpu count (no type)
# Node features that are not GPU models, so we can pick the model out of the rest.
FEAT_SKIP = re.compile(r"^(gpu|thp_|nvidia_)")
UNAVAIL = ("DOWN", "DRAIN", "DRNG", "NO_RESPOND", "FAIL", "MAINT", "INVAL", "POWER")


def gpu_type_from_features(feat: str) -> str:
    """Best-guess GPU model from a node feature list like 'gh,gpu,thp_never,...'."""
    for tok in (feat or "").split(","):
        tok = tok.strip()
        if tok and not FEAT_SKIP.match(tok):
            return tok
    return "gpu"


def gpus(field: str, default_type: str = "gpu") -> dict[str, int]:
    """type -> count from a Gres field like 'gpu:b200:8(IDX:0-7)' or 'gpu:4'.

    Bare entries with no embedded type are attributed to default_type."""
    out: dict[str, int] = defaultdict(int)
    for typ, n in GRES_RE.findall(field or ""):
        out[typ or default_type] += int(n)
    return out


def _int(kv: dict, key: str, default: int = 0) -> int:
    try:
        return int(kv.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_nodes():
    # This SLURM doesn't expose GresUsed in `scontrol show node`; used GPUs come
    # from AllocTRES (gres/gpu=N), which has no per-type breakdown. Each node here
    # has a single GPU type, so attribute all allocated GPUs to that type.
    raw = subprocess.run(["scontrol", "show", "nodes", "--oneliner"],
                         capture_output=True, text=True).stdout
    for line in raw.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        state = kv.get("State", "")
        feat = kv.get("ActiveFeatures") or kv.get("AvailableFeatures") or ""
        total = gpus(kv.get("Gres", ""), gpu_type_from_features(feat))
        m = ALLOC_RE.search(kv.get("AllocTRES", ""))
        alloc = int(m.group(1)) if m else 0
        types = list(total)
        used = {types[0]: alloc} if len(types) == 1 else {}  # single-type node
        cpu_tot = _int(kv, "CPUTot")
        cpu_used = _int(kv, "CPUAlloc")
        mem_tot = _int(kv, "RealMemory")                     # MB
        mem_used = _int(kv, "AllocMem")                      # MB
        yield {
            "name": kv.get("NodeName", "?"),
            "parts": kv.get("Partitions", "").split(","),
            "state": state,
            "unavail": any(f in state for f in UNAVAIL),
            "total": total,
            "used": used,
            "cpu_tot": cpu_tot,
            "cpu_free": max(cpu_tot - cpu_used, 0),
            "mem_tot": mem_tot,
            "mem_free": max(mem_tot - mem_used, 0),
        }


def new_agg() -> dict:
    # gpu: total/used/free/unavail; cpu/mem: total, free (all avail nodes),
    # and free_g (only on nodes that still have a free GPU of this type).
    return {"g_tot": 0, "g_used": 0, "g_free": 0, "g_un": 0,
            "c_tot": 0, "c_free": 0, "c_free_g": 0,
            "m_tot": 0, "m_free": 0, "m_free_g": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-part", action="store_true", help="break down by partition too")
    args = ap.parse_args()

    agg: dict = defaultdict(new_agg)
    for nd in parse_nodes():
        for typ, tot in nd["total"].items():
            used = nd["used"].get(typ, 0)
            free_gpu = tot - used
            keys = [(typ, p) for p in nd["parts"]] if args.by_part else [(typ, None)]
            for k in keys:
                a = agg[k]
                a["g_tot"] += tot
                a["g_used"] += used
                if nd["unavail"]:
                    a["g_un"] += free_gpu              # capacity blocked by node state
                    continue
                a["g_free"] += free_gpu                # genuinely free
                # CPU/mem headroom on this available node.
                a["c_tot"] += nd["cpu_tot"]
                a["m_tot"] += nd["mem_tot"]
                a["c_free"] += nd["cpu_free"]
                a["m_free"] += nd["mem_free"]
                if free_gpu > 0:                       # schedulable: still has a free GPU
                    a["c_free_g"] += nd["cpu_free"]
                    a["m_free_g"] += nd["mem_free"]

    def gb(mb: int) -> str:
        return f"{mb / 1024:.0f}"

    base = ("TYPE", "PART") if args.by_part else ("TYPE",)
    hdr = base + ("gpuFree", "gpuTot", "gpuUnav",
                  "cpuFree*", "cpuFree", "cpuTot",
                  "memFree*G", "memFreeG", "memTotG")
    rows = []
    for k in sorted(agg):
        a = agg[k]
        lead = (k[0], k[1] or "-") if args.by_part else (k[0],)
        rows.append(lead + (
            a["g_free"], a["g_tot"], a["g_un"],
            a["c_free_g"], a["c_free"], a["c_tot"],
            gb(a["m_free_g"]), gb(a["m_free"]), gb(a["m_tot"]),
        ))

    w = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    fmt = "  ".join("{:<%d}" % w[i] for i in range(len(hdr)))
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * x for x in w]))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    print("\n* = headroom only on nodes that still have a free GPU of that type "
          "(schedulable); the unstarred columns count all available nodes.")

    if not args.by_part:
        gT = sum(a["g_tot"] for a in agg.values())
        gU = sum(a["g_used"] for a in agg.values())
        gF = sum(a["g_free"] for a in agg.values())
        gN = sum(a["g_un"] for a in agg.values())
        cF = sum(a["c_free_g"] for a in agg.values())
        mF = sum(a["m_free_g"] for a in agg.values())
        print(f"\nALL GPUs: {gT} total, {gU} used, {gF} FREE, {gN} unavailable(down/drain)")
        print(f"On nodes with a free GPU: {cF} CPUs free, {gb(mF)}G memory free")


if __name__ == "__main__":
    main()
