"""Run the Bend analyst, apply its edit to the real checkpoint, and judge with
the checkpoint's own HF runtime. Also runs a random-direction control of the
same Frobenius norm.

Build first:  bend wsp/real.bend -o /tmp/real
Run:          python wsp/judge_real.py
"""
import subprocess
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "hf-internal-testing/tiny-random-gpt2"
PROMPTS = ["The capital of France is", "Once upon a time", "2 + 2 ="]


def analyst():
    out = subprocess.check_output(["/tmp/real"], text=True)
    edit = None
    stats = sigma = None
    for line in out.splitlines():
        if line.startswith("stats,"):
            stats = [float(x) for x in line.split(",")[1:]]
        elif line.startswith("sigma,"):
            sigma = float(line.split(",")[1])
        elif line.startswith("edit,"):
            edit = np.array([float(x) for x in line.split(",")[1:] if x], dtype=np.float32)
    return stats, sigma, edit.reshape(32, 128)


def logits(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids).logits[0, -1].float().numpy()


def compare(a, b, tok):
    top_a = np.argsort(-a)[:5]
    top_b = np.argsort(-b)[:5]
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "mean_abs": float(np.mean(np.abs(a - b))),
        "top1_changed": bool(top_a[0] != top_b[0]),
        "top5_overlap": int(len(set(top_a.tolist()) & set(top_b.tolist()))),
        "top1_before": tok.decode([int(top_a[0])]),
        "top1_after": tok.decode([int(top_b[0])]),
    }


def main():
    stats, sigma, edited = analyst()
    model = AutoModelForCausalLM.from_pretrained(MID).eval()
    tok = AutoTokenizer.from_pretrained(MID)
    w = model.transformer.h[0].mlp.c_fc.weight.data
    orig = w[0:32, 0:128].detach().cpu().numpy().copy()
    delta = edited - orig
    norm = float(np.linalg.norm(delta))
    print(f"bend_stats mean={stats[0]:.6e} std={stats[1]:.6e} rms={stats[2]:.6e} sigma={sigma:.6e}")
    print(f"delta_frobenius={norm:.6e} relative={norm/np.linalg.norm(orig):.4f}")

    base = [logits(model, tok, p) for p in PROMPTS]

    w[0:32, 0:128] = torch.tensor(edited)
    rank1 = [logits(model, tok, p) for p in PROMPTS]

    rng = np.random.default_rng(7)
    rnd = rng.standard_normal((32, 128)).astype(np.float32)
    rnd *= norm / np.linalg.norm(rnd)
    w[0:32, 0:128] = torch.tensor(orig + rnd)
    randomc = [logits(model, tok, p) for p in PROMPTS]

    w[0:32, 0:128] = torch.tensor(orig)
    rows = []
    for i, p in enumerate(PROMPTS):
        r1 = compare(base[i], rank1[i], tok)
        rc = compare(base[i], randomc[i], tok)
        rows.append((p, r1, rc))
        print(f"{p!r} rank1 max={r1['max_abs']:.3e} top1={r1['top1_changed']} top5={r1['top5_overlap']} "
              f"| random max={rc['max_abs']:.3e} top1={rc['top1_changed']} top5={rc['top5_overlap']}")
    with open("wsp/results.csv", "w") as f:
        f.write("prompt,edit,max_abs,mean_abs,top1_changed,top5_overlap,top1_before,top1_after\n")
        for p, r1, rc in rows:
            for name, r in (("rank1", r1), ("random", rc)):
                f.write(f"\"{p}\",{name},{r['max_abs']:.6e},{r['mean_abs']:.6e},{r['top1_changed']},{r['top5_overlap']},\"{r['top1_before']}\",\"{r['top1_after']}\"\n")


if __name__ == "__main__":
    main()
