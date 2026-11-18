//! The released V3 descriptor layout only; no legacy loader or arithmetic is imported.
const Codebook=enum{mcg,mul1};
pub const Layer = struct {
    gate_ptr: u64, // int64 [E]: each expert's gate trellis
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]: half-bits a value
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    count: u32, // E
    dims: u32, // D
    width: u32, // I, this rank's share of the expert width
    cb: Codebook,
    k2_gu: [2]u32, // (min, max) over gate and up
    k2_d: [2]u32,
};
