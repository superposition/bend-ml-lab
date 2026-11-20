// Device.mm is a native-only effect: the C twin uses CUDA device buffers.
// Build with `bend bench/gpu/device_mm.bend -o <out>`; the JS lane is not used.
throw new Error("Device.mm is native-only");
