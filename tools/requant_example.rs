// Drop into graft/examples/requant.rs and run: cargo run --release --example requant
use graft::codec::*;

#[derive(Clone, Copy)]
enum Kind { Q5, Q6 }

fn q(k: Kind, x: &[f32]) -> Vec<u8> {
    match k {
        Kind::Q5 => quantize_q5_k_block(x).unwrap().to_vec(),
        Kind::Q6 => quantize_q6_k_block(x).unwrap().to_vec(),
    }
}
fn dq(k: Kind, b: &[u8]) -> Vec<f32> {
    match k {
        Kind::Q5 => dequant_q5_k_block(b).unwrap(),
        Kind::Q6 => dequant_q6_k_block(b).unwrap(),
    }
}
fn gen(seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..QK_K).map(|_| {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    }).collect()
}
fn max_mean(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut mx = 0f32; let mut sum = 0f32;
    for i in 0..a.len() { let d = (a[i] - b[i]).abs(); mx = mx.max(d); sum += d; }
    (mx, sum / a.len() as f32)
}
fn main() {
    for (name, k) in [("Q5_K", Kind::Q5), ("Q6_K", Kind::Q6)] {
        let x = gen(12345);
        let q1 = q(k, &x);
        let d1 = dq(k, &q1);              // the stored values
        let q2 = q(k, &d1);               // re-encode the decoded values (factor 1.0)
        let d2 = dq(k, &q2);
        let bytes = q1.iter().zip(q2.iter()).filter(|(a,b)| a != b).count();
        let (mx, mean) = max_mean(&d1, &d2);
        let ideal: Vec<f32> = x.iter().map(|v| v * 1.1).collect();
        let q3 = q(k, &ideal);
        let d3 = dq(k, &q3);
        let (emx, emean) = max_mean(&d3, &ideal);
        println!("{name},noop_bytes={bytes}/{},noop_max={mx:.3e},noop_mean={mean:.3e},edit_max={emx:.3e},edit_mean={emean:.3e}",
                 q1.len());
    }
}
