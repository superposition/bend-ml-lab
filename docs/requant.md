# Requantization noise floor

Ticket #11: how much does a no-op (`factor = 1.0`) round-trip change a K-quant
tensor, and how does that compare to a real edit?

## Method

`tools/requant_example.rs` (drop into `graft/examples/` and run
`cargo run --release --example requant`) takes one 256-value block in
`[-1,1]`, then:

1. quantize -> `q1`; decode -> `d1` (the stored values);
2. re-quantize `d1` -> `q2`; decode -> `d2` (the no-op round-trip);
3. scale the original by 1.1, quantize -> `q3`, decode -> `d3` (a real edit).

## Result

| type | no-op bytes changed | no-op max Δ | no-op mean Δ | edit max Δ | edit mean Δ |
|---|---:|---:|---:|---:|---:|
| Q5_K | 0 / 176 | 0 | 0 | 3.42e-2 | 1.57e-2 |
| Q6_K | 0 / 210 | 0 | 0 | 1.73e-2 | 8.04e-3 |

`graft`'s encoder is **idempotent on its own blocks**: decoding and re-encoding
changes no bytes and no values. So a no-op edit in this codec is exact.

The real uncertainty is the *reference* encoder. The earlier checkpoint audit
(`aquifer/cards/graft-kquant-codec.md`) measured llama.cpp's searching encoder
against this one: re-encoding decoded values changed 5.6% of Q5_K bytes
(max |Δ| 2.9e-3) and 84% of Q6_K bytes (max |Δ| 9.2e-3). Always measure the
no-op through the encoder that will write the checkpoint, not only through this
one.

The edit error is one to two orders of magnitude larger than either no-op
figure (up to 3.4e-2 max for a 10% scale on Q5_K), so a small WSP edit is not
automatically below the quantization noise.

Raw: [`bench/einsum/requant.csv`](../bench/einsum/requant.csv).
