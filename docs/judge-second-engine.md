# Same engine vs second engine: blocked

Ticket #12 asks how much the serving engine moves a WSP score. The repo's rule
is already implemented in `docs/wsp.md`; this ticket needs a second engine
serving the *same* checkpoint.

## What is available

| engine | status |
|---|---|
| Hugging Face `transformers` 4.57.6 (WSL, torch 2.10) | used as the judge in #10 |
| llama.cpp Windows build (`C:/Users/ericm/opt/llama-prism`, `C:/Users/ericm/code/llama-cpp`) | no GGUF for the #10 checkpoint |
| vLLM | not installed |

`hf-internal-testing/tiny-random-gpt2` is an HF checkpoint; llama.cpp cannot
load it directly. A second-engine measurement needs either a GGUF conversion of
this checkpoint (and the same edit applied in both formats) or a second runtime
that loads the HF weights. Neither is provisioned.

## Protocol when available

1. Fix one edit and one evaluation set.
2. Score with engine A and engine B under the same prompts/decoding, recording
   engine, build, temperature, seed, token cap and samples per problem.
3. Report the score spread, not a single "the edit helps" number.
4. Keep the #10 random-direction control: a second engine does not remove the
   need for a same-norm control.

Status: **Blocked (external: second engine for the same checkpoint)**.
