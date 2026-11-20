# Native-Linux GPU baseline: blocked

Ticket #17 needs a host where the GPU exposes `concurrentManagedAccess=1`.
This machine does not have one available:

- WSL2 reports `cma=0` (CUDA 12.8, RTX 4090). This is a platform property, not
  a driver setting.
- The Jetson Orin NX is the intended native-Linux host, but it is unreachable:
  `ssh pinkie` (10.0.0.180), `192.168.55.1`, and `pinkie-router-wifi`
  (10.0.0.34) all time out; port 22 is closed on the Wi-Fi address.
- No Linux partition, GPU-passthrough VM, or cloud GPU credentials are
  configured on this workstation.

The experiment is ready to run when a host exists:

```bash
# on native Linux, with stock Bend 2.0.40
bend bench/einsum/gpu_tile.bend -o /tmp/tile
/tmp/tile +RTS --gpu 512MB
```

Record: `cma` (`cudaDevAttrConcurrentManagedAccess`), GPU, driver, CUDA,
kernel ms, and the `gpu.csv` row. The WSL2 control is **95 ms** (tree) and
**5.31 ms** (contiguous device buffers, `docs/contiguous.md`). The question is
how much of the 95 ms is WSL2 migration and how much is the tree; the
standalone CUDA decomposition already estimates 67 ms migration / 27 ms tree.

Status: **Blocked (external hardware)**. Not a code task.
