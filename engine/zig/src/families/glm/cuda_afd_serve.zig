//! One served stream on the AFD attention node (qualification_4be15bbf): cuda_serve.zig's serve calls (qualification_qualification_44ba554e.8's) over afd_engine.Attn, so
//! a server's rounds make the launches and the exchange messages tf-glm-afd attn makes for the same prompt: the prompt by
//! Attn.prefill (decode.prefill under AFD, each chunk begun MODE_CHUNK), then python's serial_decode one step a call
//! (Attn.step: the forward begun eager, synced and checked, then the greedy draw and the commit). Greedy, drafts off;
//! each prefill starts the sequence afresh (State.reset). A failed exchange fails the call that met it, and Attn's
//! failure is sticky: every later call refuses before any launch (the halves are tf-glm-afd expert processes).
const std = @import("std");
const Attn = @import("cuda_afd_engine.zig").Attn;
const Fwd = @import("cuda_fwd.zig").Fwd;

/// The serve calls (native/glm_cuda_lanes.zig's `Steps`) over the attention node `at` and its Fwd.
pub const Serve = struct {
    at: *Attn,
    f: *Fwd,
    /// rows the sequence holds: its prompt, then each fed token (the State's position)
    len: usize = 0,
    open: bool = false,
    deferred: bool = false,

    pub fn installStructured(s: *Serve, endpoint: @import("lanes").structured_control.Endpoint) !void {
        try s.checkMasked();
        if (!@import("builtin").is_test or s.f.gpu != null or s.f.rec == null or s.at.a.wire != null) return error.IncompleteMaskedAdapter;
        if (s.at.structured_control != null) return error.OperationInProgress;
        s.at.structured_control = endpoint;
        s.at.deferred_logits = null;
    }
    pub fn clearStructured(s: *Serve) void {
        if (s.at.structured_control != null) {
            s.at.structured_control = null;
            s.at.deferred_logits = null;
            s.at.defer_first = false;
            s.at.pending = false;
            s.at.e.st.pos = 0;
            s.len = 0;
            s.open = false;
            s.deferred = false;
        }
    }
    pub fn maskedWidth(s: *Serve) usize { return s.at.e.m.vocab; }
    pub fn maskedWords(s: *Serve) usize {
        return (s.at.e.m.vocab + 31) / 32;
    }
    pub fn checkMasked(s: *Serve) !void {
        try @import("cuda_sample.zig").maskedReady(s.f, &s.at.e.smp);
    }
    pub fn prefillToLogits(s: *Serve, ids: []const u32) !void {
        try s.checkMasked();
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len + 1 > s.at.e.st.capacity) return error.PastCapacity;
        s.open = false;
        s.deferred = false;
        try s.at.prefillToLogits(s.f, ids);
        s.len = ids.len;
        s.open = true;
        s.deferred = true;
    }
    pub fn pickFirst(s: *Serve, mask: []const u32) !u32 {
        if (!s.open or !s.deferred) return error.NoDeferredFirst;
        const next = try s.at.pickFirst(s.f, mask);
        s.deferred = false;
        return next;
    }
    pub fn stepMasked(s: *Serve, token: u32, mask: []const u32) !u32 {
        if (!s.open or s.deferred) return error.NoSequence;
        if (s.len + 1 > s.at.e.st.capacity) return error.PastCapacity;
        const next = try s.at.stepMasked(s.f, token, mask);
        s.len += 1;
        return next;
    }

    /// decode.prefill under AFD: the prompt in prefill_rows chunks through both halves, committed; its greedy first token.
    pub fn prefill(s: *Serve, ids: []const u32) !u32 {
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len + 1 > s.at.e.st.capacity) return error.PastCapacity;
        s.open = false;
        const first = try s.at.prefill(s.f, ids);
        s.len = ids.len;
        s.open = true;
        return first;
    }

    /// serial_decode's step under AFD: `token` fed at the sequence's end through both halves, its greedy next, committed.
    pub fn step(s: *Serve, token: u32) !u32 {
        if (!s.open) return error.NoSequence;
        if (s.len + 1 > s.at.e.st.capacity) return error.PastCapacity;
        const next = try s.at.step(s.f, token);
        s.len += 1;
        return next;
    }

    /// The stream ended: the next prefill resets the State, so nothing is freed here.
    pub fn release(s: *Serve) void {
        s.deferred = false;
        s.at.deferred_logits = null;
        s.open = false;
    }
};
