// Node twin for bench/einsum/main.bend. Run: node main.js 512
const n = Number(process.argv[2] || 512);
const cells = n * n;
const a = new Float32Array(cells);
const b = new Float32Array(cells);
const c = new Float32Array(cells);

function prng(x) {
  x ^= x << 13; x >>>= 0;
  x ^= x >>> 17; x >>>= 0;
  x ^= x << 5; x >>>= 0;
  return x >>> 0;
}
function gen(arr, seed) {
  for (let i = 0; i < cells; i++) {
    seed = prng(seed);
    arr[i] = (seed >>> 8) / 16777216;
  }
}
function checksum(arr) {
  let s = 0;
  for (let i = 0; i < cells; i++) s += arr[i];
  return s;
}
let t0 = Date.now();
gen(a, 7); gen(b, 11);
let t1 = Date.now();
for (let i = 0; i < n; i++) {
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += a[i * n + k] * b[k * n + j];
    c[i * n + j] = s;
  }
}
let t2 = Date.now();
console.log(`JS,seq,gen,${checksum(a) + checksum(b)},${(t1 - t0).toFixed(3)}`);
console.log(`JS,seq,matmul,${checksum(c)},${(t2 - t1).toFixed(3)}`);
