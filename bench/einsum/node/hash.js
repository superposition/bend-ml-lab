const n=64, cells=n*n;
const a=new Float32Array(cells), b=new Float32Array(cells), c=new Float32Array(cells);
function prng(x){x^=x<<13;x>>>=0;x^=x>>>17;x>>>=0;x^=x<<5;x>>>=0;return x>>>0;}
function gen(arr,s){for(let i=0;i<cells;i++){s=prng(s);arr[i]=(s>>>8)/16777216;}}
function gen2(arr,s){for(let i=0;i<cells;i++){s=prng(s);arr[i]=2*((s>>>8)/16777216)-1;}}
function hash(a){const u=new Uint32Array(a.buffer);let h=2166136261;for(let i=0;i<cells;i++)h=Math.imul(h^u[i],16777619)>>>0;return h;}
gen(a,7);gen(b,11);
for(let i=0;i<n;i++)for(let j=0;j<n;j++){let s=0;for(let k=0;k<n;k++)s=Math.fround(s+Math.fround(a[i*n+k]*b[k*n+j]));c[i*n+j]=s;}
console.log(`seq,${hash(c)}`);
gen2(a,13);gen2(b,17);
for(let i=0;i<n;i++)for(let j=0;j<n;j++){let s=0;for(let k=0;k<n;k++)s=Math.fround(s+Math.fround(a[i*n+k]*b[k*n+j]));c[i*n+j]=s;}
console.log(`neg,${hash(c)}`);
