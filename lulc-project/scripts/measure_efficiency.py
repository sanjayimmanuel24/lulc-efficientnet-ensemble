"""
Computational cost measurements for docs/BLUEPRINT.md Section 13.

The paper is framed around efficiency and C3 (ECA) was chosen specifically on
parameter cost, but nothing in the repo ever measured either. This produces the
numbers those claims need -- including the uncomfortable one: the proposed model
runs TWO EfficientNet-B0 backbones, so it costs roughly 2x a single B0. That has
to be stated and argued, not left for a reviewer to compute.

    python scripts/measure_efficiency.py
    python scripts/measure_efficiency.py --image-size 224 --batch-size 32 --runs 50

Writes results/efficiency.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import spectral_in_chans
from src.models.ensemble import DualBranchEfficientNet, _Branch

PROJECT_ROOT = Path(__file__).parent.parent
OUT = PROJECT_ROOT / "results" / "efficiency.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--runs", type=int, default=30, help="timed forward passes (after warmup)")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="cpu matters for the edge-deployment framing: a drone or ground station "
                        "may have no GPU at all, and that is the setting the paper argues for.")
    p.add_argument("--out", default=None, help="output filename under results/ (default efficiency.json)")
    return p.parse_args()


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, inputs):
    """
    MACs via torch's built-in FlopCounterMode (no thop/fvcore dependency).

    Note torch reports multiply-accumulates; papers often quote FLOPs = 2 x MACs.
    Both are emitted so the paper can state which convention it uses.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    counter = FlopCounterMode(display=False)
    with counter:
        model(*inputs)
    return counter.get_total_flops()


def time_forward(model, inputs, runs, device):
    with torch.no_grad():
        for _ in range(5):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / runs


def peak_vram(model, inputs, device):
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    S, B = args.image_size, args.batch_size
    print("device={}  image_size={}  batch_size={}  runs={}".format(device, S, B, args.runs))

    rows = {}
    configs = [
        ("single_b0_rgb_only", None, "eca"),         # the efficiency reference point
        ("dual_branch_eca", "rgb_plus_indices", "eca"),
        ("dual_branch_no_attention", "rgb_plus_indices", "none"),
        ("dual_branch_nonrgb_eca", "nonrgb", "eca"),
    ]

    for name, mode, attention in configs:
        if mode is None:
            model = _Branch(in_chans=3, num_classes=args.num_classes,
                            pretrained=False, attention=attention).to(device).eval()
            inputs = (torch.randn(B, 3, S, S, device=device),)
        else:
            chans = spectral_in_chans(mode, 1)
            model = DualBranchEfficientNet(num_classes=args.num_classes, pretrained=False,
                                            spectral_in_chans=chans,
                                            attention=attention).to(device).eval()
            inputs = (torch.randn(B, 3, S, S, device=device),
                      torch.randn(B, chans, S, S, device=device))

        total, trainable = count_params(model)
        macs = count_flops(model, inputs)
        latency = time_forward(model, inputs, args.runs, device)
        vram = peak_vram(model, inputs, device)

        rows[name] = {
            "params_total": total,
            "params_trainable": trainable,
            "params_millions": round(total / 1e6, 3),
            "model_size_mb_fp32": round(total * 4 / 1e6, 2),
            "macs": macs,
            "gmacs_per_batch": round(macs / 1e9, 3) if macs else None,
            "gmacs_per_image": round(macs / 1e9 / B, 4) if macs else None,
            "latency_s_per_batch": round(latency, 5),
            "latency_ms_per_image": round(latency / B * 1000, 3),
            "throughput_img_per_s": round(B / latency, 1),
            "peak_vram_mb": round(vram / 1e6, 1) if vram else None,
        }
        print("  {:<26} {:>7.3f}M params  {:>8}  {:>7.2f} ms/img  {:>7.1f} img/s".format(
            name, total / 1e6,
            "{:.3f} GMACs".format(macs / 1e9) if macs else "n/a",
            latency / B * 1000, B / latency))
        del model, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The two numbers the paper's efficiency framing actually turns on.
    single = rows["single_b0_rgb_only"]["params_total"]
    dual = rows["dual_branch_eca"]["params_total"]
    no_attn = rows["dual_branch_no_attention"]["params_total"]
    summary = {
        "dual_vs_single_param_ratio": round(dual / single, 3),
        "eca_params_total": dual - no_attn,
        "eca_params_as_fraction_of_model": (dual - no_attn) / dual,
        "dual_vs_single_latency_ratio": round(
            rows["dual_branch_eca"]["latency_s_per_batch"]
            / rows["single_b0_rgb_only"]["latency_s_per_batch"], 3),
    }
    print("")
    print("dual-branch costs {:.2f}x the parameters of a single EfficientNet-B0".format(
        summary["dual_vs_single_param_ratio"]))
    print("ECA adds {} parameters total ({:.2e} of the model) -- C3's efficiency claim".format(
        summary["eca_params_total"], summary["eca_params_as_fraction_of_model"]))
    print("dual-branch inference is {:.2f}x slower per batch".format(
        summary["dual_vs_single_latency_ratio"]))

    out_path = OUT if not args.out else OUT.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"device": str(device), "image_size": S, "batch_size": B,
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "note": "macs are multiply-accumulates; FLOPs = 2 x MACs under the common convention",
                   "variants": rows, "summary": summary}, f, indent=2)
    print("")
    print("Saved " + str(out_path))


if __name__ == "__main__":
    main()
