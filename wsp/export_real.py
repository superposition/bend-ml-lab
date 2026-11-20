"""Export the c_fc.weight [0:32,0:128] slice of tiny-random-gpt2 as Bend source."""
import torch
from transformers import AutoModelForCausalLM

MID = "hf-internal-testing/tiny-random-gpt2"
m = AutoModelForCausalLM.from_pretrained(MID)
w = m.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy()
sl = w[0:32, 0:128]
vals = []
for v in sl.reshape(-1):
    x = float(v)
    if x < 0:
        vals.append(f"F32.neg({abs(x):.9g})")
    else:
        vals.append(f"{x:.9g}")
body = "[" + ",".join(vals) + ",]"
src = f'''import Base

# Exported from {MID} transformer.h.0.mlp.c_fc.weight[0:32,0:128].
# Real checkpoint values; 32x128, row-major.
def tensor() -> List<&2, F32>:
  {body}
'''
open("wsp/real_tensor.bend", "w").write(src)
print("wrote", len(vals), "values", sl.shape, "min", sl.min(), "max", sl.max())
