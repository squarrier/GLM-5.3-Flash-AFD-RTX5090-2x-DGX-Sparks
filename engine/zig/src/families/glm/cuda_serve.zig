//! One served stream on the CUDA Engine (qualification_qualification_44ba554e.8 day 1): the prompt by decode.prefill, then python's serial_decode one
//! step a call (forward, greedy, commit), so a server's rounds make the launches tf-glm-forward's `first` and
//! Engine.serialDecode make, which qualification_66fa03c8/qualification_fefbc9d8 checked against python-0.6.6 on two Sparks. Greedy; with drafts on (qualification_qualification_44ba554e.8
//! day 3) the prefill absorbs into the MTP head and native/glm_cuda_drafts.zig runs qualification_qualification_86ea5170's loops on `e` and `f`; each
//! prefill starts the sequence afresh (State.reset).
const std = @import("std");
const eng = @import("cuda_engine.zig");
const cc = @import("cuda_compute.zig");
const sample = @import("cuda_sample.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

/// The serve calls over an Engine of type `E` (eng.Engine; a test's Engine of markers).
pub fn ServeOf(comptime E: type) type {
    return struct {
        const Self = @This();

        e: *E,
        f: *Fwd,
        /// rows the sequence holds: its prompt, then each fed token (the State's position)
        len: usize = 0,
        open: bool = false,
        /// drafts on (qualification_qualification_44ba554e.8 day 3): the prefill absorbs into the MTP head, python's decode.prefill(mtp=use_mtp)
        absorb: bool = false,
        deferred: bool = false,

        pub fn installStructured(s: *Self, endpoint: @import("lanes").structured_control.Endpoint) !void {
            try s.checkMasked();
            if (!@import("builtin").is_test or s.f.gpu != null or s.f.rec == null) return error.IncompleteMaskedAdapter;
            if (s.e.structured_control != null) return error.OperationInProgress;
            s.e.structured_control = endpoint;
            s.e.structured_logits_valid = false;
            s.e.smp.masked_cpu.?.endpoint = endpoint;
            s.e.smp.masked_cpu.?.local_rank = s.f.m.rank;
        }
        pub fn clearStructured(s: *Self) void {
            if (s.e.structured_control != null) {
                s.e.structured_control = null;
                s.e.structured_logits_valid = false;
                s.e.smp.masked_cpu.?.endpoint = null;
                s.e.smp.masked_cpu.?.local_rank = null;
                s.e.smp.masked_cpu.?.permit = null;
                s.e.st.pos = 0;
                s.len = 0;
                s.open = false;
                s.deferred = false;
            }
        }
        pub fn maskedWidth(s: *Self) usize { return s.e.m.vocab; }
        pub fn maskedWords(s: *Self) usize {
            return (s.e.m.vocab + 31) / 32;
        }
        pub fn checkMasked(s: *Self) !void {
            try sample.maskedReady(s.f, &s.e.smp);
        }
        pub fn prefillToLogits(s: *Self, ids: []const u32) !void {
            try s.checkMasked();
            if (ids.len == 0) return error.EmptyPrompt;
            if (ids.len + 1 > s.e.st.capacity) return error.PastCapacity;
            s.open = false;
            s.deferred = false;
            try s.e.prefillToLogits(s.f, ids);
            s.len = ids.len;
            s.open = true;
            s.deferred = true;
        }
        pub fn pickFirst(s: *Self, mask: []const u32) !u32 {
            if (!s.open or !s.deferred) return error.NoDeferredFirst;
            const next = try s.e.pickFirst(s.f, mask);
            s.deferred = false;
            return next;
        }
        pub fn stepMasked(s: *Self, token: u32, mask: []const u32) !u32 {
            if (!s.open or s.deferred) return error.NoSequence;
            if (s.len + 1 > s.e.st.capacity) return error.PastCapacity;
            const next = try s.e.stepMasked(s.f, token, mask);
            s.len += 1;
            return next;
        }

        /// decode.prefill (absorbing into the MTP head with `absorb`): the prompt in prefill_rows chunks, committed,
        /// its greedy first token.
        pub fn prefill(s: *Self, ids: []const u32) !u32 {
            if (ids.len == 0) return error.EmptyPrompt;
            if (ids.len + 1 > s.e.st.capacity) return error.PastCapacity;
            s.open = false;
            const first = try s.e.prefill(s.f, ids, s.absorb);
            s.len = ids.len;
            s.open = true;
            return first;
        }

        /// serial_decode's step: `token` fed at the sequence's end, its row's greedy token drawn, the row committed.
        pub fn step(s: *Self, token: u32) !u32 {
            if (!s.open) return error.NoSequence;
            if (s.len + 1 > s.e.st.capacity) return error.PastCapacity;
            const logits = try s.e.forward(s.f, &.{token});
            var out: [1]u32 = undefined;
            try sample.greedy(s.f, &s.e.smp, logits, &out);
            try cc.commit(s.f, &s.e.st, &s.e.buf, 1, 1);
            s.len += 1;
            return out[0];
        }

        /// The stream ended: the next prefill resets the State, so nothing is freed here.
        pub fn release(s: *Self) void {
            s.deferred = false;
            s.open = false;
        }
    };
}

/// The server's serve calls: python's Engine with the MTP layer over lane A's blocks (unused while drafts are off).
pub const Serve = ServeOf(eng.Engine);
