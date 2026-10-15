//! The AFD attention node's engine at world 1 (qualification_25364d42 PLAN §1.1 A1-A16, §4): python 2.15's Engine.forward and
//! decode.prefill with `w.afd` (decode.py:172-174, 555-556), over B3's Engine (cuda_engine.zig: stage, compute, the
//! greedy sample, commit) with qualification_9051d0c1's provider as Fwd.moe. A decode step is Afd.begin (eager), stage and compute, one
//! stream sync and Node.check, and only then the sample and the commit; a prompt runs in chunks of prefill_rows, each
//! begun MODE_CHUNK, its first token sampled after the last chunk's sync and check. A failed exchange fails the step's
//! request with Node.failure()'s reason and its logits are never read; every later step refuses before any launch (the
//! failure is sticky) and the process goes on. TF_AFD_CHECK (`check`): each forward's layers hashed after its sync.
const std = @import("std");
const cuda = @import("cuda");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const cc = @import("cuda_compute.zig");
const sample = @import("cuda_sample.zig");
const eng = @import("cuda_engine.zig");
const dump = @import("cuda_dump.zig");
const layer = @import("cuda_layer.zig");
const afd = @import("cuda_afd_attn.zig");
const audit = @import("cuda_afd_audit.zig");
const amtp = @import("cuda_afd_mtp.zig");
const prompt = @import("cuda_prompt.zig");
const pairs = @import("cuda_afd_pairs.zig");
const graphs = @import("cuda_afd_graphs.zig");
const dsa = @import("cuda_dsa.zig");
const cmtp = @import("cuda_mtp.zig");
const lanes = @import("cuda_afd_lanes.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const control = @import("lanes").structured_control;

const xchg = cuda.xchg;
const T = trace.T;

/// TF_AFD_CHECK's attention half: each checked forward's layers (the provider's ring, hashed) go to `forward`.
pub const Check = struct {
    ctx: *anyopaque,
    forward: *const fn (ctx: *anyopaque, seq: u32, R: usize, layers: []const audit.Layer) anyerror!void,
    /// one forward's records (the model's MoE layers)
    layers: []audit.Layer,
};

/// The attention node: B3's Engine and the exchange's provider; `failed` keeps the first failure's reason.
pub const Attn = struct {
    e: *eng.Engine,
    a: *afd.Afd,
    failed: ?xchg.Failure = null,
    check: ?Check = null,
    /// a forward whose layers the audit has not hashed yet (a prompt chunk's: python hashes at the next begin)
    pending: bool = false,
    forwards: u64 = 0,
    /// qualification_21e21bc0: prefill absorbs each chunk's next tokens into the MTP head's cache (a drafted request; TF_AFD_MTP=1)
    absorb: bool = false,
    /// qualification_0f991257 optimized prompt pairs, unchanged default off.
    pairs: ?*pairs.Pairs = null,
    /// qualification_f16009a8 decode graphs, unchanged default off.
    graphs: ?*graphs.Graphs = null,
    /// qualification_9e49592e optimized prompt lanes, unchanged default off.
    lanes: ?*lanes.Lanes = null,
    defer_first: bool = false,
    deferred_logits: ?T = null,
    structured_control: ?control.Endpoint = null,

    fn descriptor(x: *Attn, kind: control.Kind, tokens: []const u32, mask: []const u32, action: control.Action) control.Descriptor {
        const p = x.structured_control.?;
        return .{ .kind = kind, .epoch = p.session.epoch, .peer_count = p.session.peer_count,
            .position = x.e.st.pos + @intFromBool(kind == .step), .rows = 1, .tokens = tokens, .mask = mask,
            .vocab_width = @intCast(x.e.m.vocab), .active = mask.len != 0, .action = action };
    }
    fn done(p: control.Endpoint, permit: control.Permit, completion: control.Completion) !control.Completion {
        const result = try p.after(permit, completion);
        switch (result.status) {
            .ok => {}, .cancelled => return error.Cancelled, .empty_mask => return error.EmptyMask,
            .request_error => return error.StructuredRequestError, .poison => return error.ControlPoison,
        }
        return result;
    }
    fn select(x: *Attn, f: *Fwd, logits: T, mask: []const u32, permit: ?control.Permit, position: usize) !u32 {
        var out: [1]u32 = undefined;
        // structured_users: a sampled request (TF_GLM_SAMPLING=1) draws among the mask's tokens by the nucleus, the mask applied to
        // the widened row before the temperature and the cuts, keyed at the plain sampled path's position
        const sampled = if (x.e.sampling) |s| s.temperature > 0 else false;
        (if (sampled) x.e.sampleMasked(f, logits, position, &out, mask) else sample.greedyMasked(f, &x.e.smp, logits, &out, mask)) catch |err| {
            if (permit) |q| {
                if (err != error.EmptyMask) {
                    _ = x.structured_control.?.quiesce(.poison) catch {};
                    return error.ControlPoison;
                }
                _ = try done(x.structured_control.?, q, .{ .logits_valid = true });
            }
            return err;
        };
        if (permit) |q| {
            const result = try done(x.structured_control.?, q, .{ .logits_valid = true,
                .candidate = .{ .valid = true, .global_id = out[0], .value_f32_bits = @bitCast(@as(f32, 0)) } });
            if (!result.candidate.valid or result.candidate.global_id != out[0]) return error.ControlPoison;
        }
        return out[0];
    }

    pub fn committedFault(x: *Attn, original: anyerror) anyerror {
        const p = x.structured_control orelse return original;
        x.deferred_logits = null; x.e.structured_logits_valid = false; x.pending = false;
        _ = p.quiesce(.poison) catch {};
        return error.ControlPoison;
    }
    pub fn prefillToLogits(x: *Attn, f: *Fwd, ids: []const u32) !void {
        try sample.maskedReady(f, &x.e.smp);
        const saved = x.absorb;
        x.absorb = false;
        x.defer_first = true;
        x.deferred_logits = null;
        defer {
            x.absorb = saved;
            x.defer_first = false;
        }
        errdefer x.deferred_logits = null;
        _ = try x.prefill(f, ids);
        if (x.deferred_logits == null) return error.NoDeferredFirst;
    }
    pub fn pickFirst(x: *Attn, f: *Fwd, mask: []const u32) !u32 {
        try sample.mask_cpu.shape(x.e.m.vocab, mask);
        const logits = x.deferred_logits orelse return error.NoDeferredFirst;
        errdefer x.deferred_logits = null;
        const permit: ?control.Permit = if (x.structured_control) |p| try p.before(x.descriptor(.first, &.{}, mask, .select)) else null;
        const next = try x.select(f, logits, mask, permit, x.e.st.pos);
        x.deferred_logits = null;
        return next;
    }
    pub fn stepMasked(x: *Attn, f: *Fwd, token: u32, mask: []const u32) !u32 {
        try sample.mask_cpu.shape(x.e.m.vocab, mask);
        try sample.maskedReady(f, &x.e.smp);
        try sample.maskedPrecheck(&x.e.smp, mask); // grammar_mask: an empty device mask ends here, before any forward
        const permit: ?control.Permit = if (x.structured_control) |p| try p.before(x.descriptor(.step, &.{token}, mask, .none)) else null;
        const logits = x.forward(f, &.{token}) catch |err| return x.committedFault(err);
        const pick: ?control.Permit = if (permit) |q| blk: {
            _ = try done(x.structured_control.?, q, .{ .logits_valid = true });
            break :blk try x.structured_control.?.before(x.descriptor(.step, &.{token}, mask, .select));
        } else null;
        const next = try x.select(f, logits, mask, pick, x.e.st.pos + 1);
        cc.commit(f, &x.e.st, &x.e.buf, 1, 1) catch |err| return x.committedFault(err);
        return next;
    }

    /// python 2.15's attention Engine: world 1, MTP refused under AFD (engine.py:499-502) unless the halves serve the
    /// MTP layer's experts (qualification_21e21bc0: TF_AFD_MTP=1, the provider's `mtp`), the provider's model.
    pub fn init(e: *eng.Engine, a: *afd.Afd) !Attn {
        if (e.m.world != 1) return error.NotWorld1;
        if (e.mbuf != null and !a.mtp) return error.MtpUnderAfd;
        if (a.mtp and e.mbuf == null) return error.NoMtp;
        if (a.m != e.m) return error.OtherModel;
        if (a.rows < e.buf.rows or a.rows < e.prefill_rows) return error.Rows;
        return .{ .e = e, .a = a };
    }

    /// The exchange's failure (null while it is alive), and this node's latch: the reason the requests fail with.
    pub fn failure(x: *Attn) ?xchg.Failure {
        if (x.failed != null) return x.failed;
        const w = x.a.wire orelse return null;
        x.failed = w.node.failure();
        return x.failed;
    }

    /// Refused before any launch once the exchange failed (it is sticky) or this node died (python's `dead`).
    pub fn alive(x: *Attn) !void {
        if (x.a.wire) |w| {
            w.node.check() catch |err| {
                _ = x.failure();
                return err;
            };
        }
        if (x.a.dead) {
            _ = x.failure();
            return error.ExpertFailed;
        }
    }

    /// afd.begin before a forward's launches (and the provider as Fwd.moe): refused at once once the exchange failed.
    fn begin(x: *Attn, f: *Fwd, R: usize, mode: afd.Mode) !void {
        return x.beginOf(f, R, mode, false);
    }

    /// `begin` for the backbone's positions, or (`mtp`, qualification_21e21bc0) the MTP head's slice.
    fn beginOf(x: *Attn, f: *Fwd, R: usize, mode: afd.Mode, mtp: bool) !void {
        try x.finish(f); // python's _finish_check at the next begin: the last checked forward's sync and hashes
        f.moe = x.a.provider();
        const s = if (f.gpu) |g| g.stream else null;
        (if (mtp) x.a.beginMtp(R, mode, s) else x.a.begin(R, mode, s)) catch |err| {
            if (err == error.ExpertFailed or err == error.OutOfStep) _ = x.failure();
            return err;
        };
        x.forwards += 1;
    }

    /// The step's one stream sync, then Node.check: an exchange that failed fails this step's request.
    pub fn settle(f: *Fwd, x: *Attn) !void {
        if (f.gpu) |g| try g.stream.synchronize();
        const w = x.a.wire orelse return;
        w.node.check() catch |err| {
            _ = x.failure();
            return err;
        };
    }

    /// A forward's launches are all queued: with the audit on, its layers wait for `finish`.
    pub fn launched(x: *Attn) void {
        x.pending = x.check != null and x.a.ring != null;
    }

    /// The audit of the forward last launched: its sync and check, then its layers hashed (wider forwards than the
    /// ring's rows are outside the window, as python's check covers only the rows it was given).
    pub fn finish(x: *Attn, f: *Fwd) !void {
        if (!x.pending) return;
        x.pending = false;
        try settle(f, x);
        const c = x.check.?;
        const r = x.a.ring.?;
        const R = x.a.cur;
        if (R > r.rows) return;
        const n = 4 * R * x.a.layout.d;
        for (c.layers[0..x.a.count], x.a.first..) |*l, pos| { // the forward's positions (qualification_21e21bc0: an MTP slice's one)
            const s = r.at(pos);
            l.* = audit.attnLayer(x.a.layout, R, s[0], s[1][0..n], s[2][0..n]);
        }
        try c.forward(c.ctx, x.a.seq, R, c.layers[0..x.a.count]);
    }

    /// Engine.forward under AFD (decode.py:172-174): begin (eager), stage and compute, the sync and the check; the
    /// logits [R, V] are valid only when it returns. qualification_f16009a8: the compute through the graphs when they are on.
    pub fn forward(x: *Attn, f: *Fwd, tokens: []const u32) !T {
        try x.begin(f, tokens.len, .eager);
        const logits = if (x.graphable(tokens.len)) try x.graphed(f, tokens) else try x.e.forward(f, tokens);
        try settle(f, x);
        x.launched();
        try x.finish(f);
        return logits;
    }

    /// qualification_f16009a8: a window the graphs take: no recorder tap, no profile split, no pair, at most the decode Buffers' rows.
    fn graphable(x: *const Attn, R: usize) bool {
        return x.graphs != null and x.e.tap == null and x.a.mark == null and x.a.pair == null and R <= @min(x.e.buf.rows, graphs.max_rows);
    }

    /// Engine.forward with the graphs: the stage eager, then the compute of the window's key, replayed or captured.
    fn graphed(x: *Attn, f: *Fwd, tokens: []const u32) !T {
        const e = x.e;
        const R = try cc.stage(f, &e.st, &e.buf, tokens);
        const opt: layer.Opt = .{ .nch = cc.chunksFor(&e.st, R), .host_pos = e.st.pos, .chip = e.chip };
        var sh: dsa.Shape = .{}; // every DSA layer's choices are the first's (one State, one Buffers, one position)
        for (e.w.layers) |*l| if (l.dsa != null) {
            sh = try dsa.shape(e.m, l, &e.st, &e.buf, R, opt);
            break;
        };
        var body: Body = .{ .x = x, .f = f, .b = &e.buf, .R = R, .opt = opt, .mtp = false };
        _ = try x.graphs.?.run(f.rec, .{ .kind = .backbone, .rows = R, .first = x.a.first, .count = x.a.count, .parity = e.st.parity, .dense = sh.dense, .nch = sh.nch, .sparse = sh.sparse, .np = sh.np }, &body);
        return body.out.?;
    }

    /// qualification_f16009a8: an MTP step through the graphs (decode's absorb and drafts: eager mode, no prompt chunk's context).
    fn mtpGraphed(x: *Attn, f: *Fwd, b: *bufs.Buffers, next: []const u32, hidden: T) !T {
        const e = x.e;
        const n = try cmtp.stage(f, &e.st, b, next, hidden);
        const opt = amtp.options(&e.st, n, e.chip, null);
        const head = e.w.mtp orelse return error.NoMtp;
        const sh = try dsa.shape(e.m, &head.layer, &e.st, b, n, opt);
        var body: Body = .{ .x = x, .f = f, .b = b, .R = n, .opt = opt, .mtp = true };
        _ = try x.graphs.?.run(f.rec, .{ .kind = .mtp, .rows = n, .first = x.a.first, .count = x.a.count, .zero_first = b.zero_first, .dense = sh.dense, .nch = sh.nch, .sparse = sh.sparse, .np = sh.np }, &body);
        return body.out.?;
    }

    /// One greedy decode step from `tok`: its next token, then the commit (python serial_decode's step under AFD).
    pub fn step(x: *Attn, f: *Fwd, tok: u32) !u32 {
        const logits = try x.forward(f, &.{tok});
        var out: [1]u32 = .{0}; // recorded runs leave it (nothing ran)
        if (x.e.sampling) |s| {
            if (s.temperature > 0) try x.e.sampleRows(f, logits, x.e.st.pos + 1, &out, null) else try sample.greedy(f, &x.e.smp, logits, &out);
        } else try sample.greedy(f, &x.e.smp, logits, &out);
        try cc.commit(f, &x.e.st, &x.e.buf, 1, 1);
        return out[0];
    }

    /// decode.serial_decode under AFD: from `pending` one token a step into `out`, to `count` or an eos (stop_eos).
    pub fn decode(x: *Attn, f: *Fwd, pending: u32, count: usize, out: []u32, stop_eos: bool) ![]u32 {
        if (count == 0 or count > out.len) return error.Count;
        out[0] = pending;
        var n: usize = 1;
        while (n < count and !(stop_eos and std.mem.indexOfScalar(u32, x.e.m.eos[0..x.e.m.eos_n], out[n - 1]) != null)) : (n += 1) {
            out[n] = try x.step(f, out[n - 1]);
        }
        return out[0..n];
    }

    /// decode.prefill under AFD (decode.py:555-556): each chunk begun MODE_CHUNK, staged, computed (B3's chunk path),
    /// its last row's logits and hidden row cloned, committed; then the sync and the check before the first token.
    /// With `absorb` (qualification_21e21bc0) each chunk's next tokens go through the MTP head first (Engine.prefill's absorb, under AFD).
    /// With `pairs` (qualification_0f991257, TF_AFD_PAIRS=1) a prompt of 2+ chunks runs them in pairs (cuda_afd_pairs.zig): the same
    /// chunks, launches and exchanges, interleaved by layer (no recorder tap: a tapped prefill stays chunk by chunk).
    /// With `lanes` (qualification_9e49592e, TF_AFD_LANES=N) such a prompt runs in lanes of up to N chunks instead (cuda_afd_lanes.zig).
    pub fn prefill(x: *Attn, f: *Fwd, ids: []const u32) !u32 {
        if (x.lanes) |l| if (ids.len > x.e.prefill_rows and x.e.tap == null) return l.prefill(x, f, ids);
        if (x.pairs) |p| if (ids.len > x.e.prefill_rows and x.e.tap == null) return p.prefill(x, f, ids);
        if (ids.len == 0) return error.EmptyPrompt;
        try x.alive(); // before State.reset's launches
        const e = x.e;
        const reset: ?control.Permit = if (x.structured_control) |p| blk: {
            try p.open();
            break :blk try p.before(.{ .kind = .reset, .epoch = p.session.epoch,
                .peer_count = p.session.peer_count, .total_prompt_rows = ids.len });
        } else null;
        cc.reset(f, &e.st) catch |err| return x.committedFault(err);
        if (reset) |q| _ = try done(x.structured_control.?, q, .{});
        var last: ?T = null;
        var start: usize = 0;
        while (start < ids.len) : (start += e.prefill_rows) last = try x.chunk(f, ids, start);
        return x.firstToken(f, last.?);
    }

    /// prefill_interleave: `prefill`'s plain path a step at a time: its preamble with the first, one chunk a call from `start`,
    /// the first token with the last (null while chunks remain); the same calls in the same order as the whole pass.
    pub fn promptStep(x: *Attn, f: *Fwd, ids: []const u32, start: *usize, first: bool) !?u32 {
        const e = x.e;
        if (first) {
            if (ids.len == 0) return error.EmptyPrompt;
            try x.alive(); // before State.reset's launches
            const reset: ?control.Permit = if (x.structured_control) |p| blk: {
                try p.open();
                break :blk try p.before(.{ .kind = .reset, .epoch = p.session.epoch,
                    .peer_count = p.session.peer_count, .total_prompt_rows = ids.len });
            } else null;
            cc.reset(f, &e.st) catch |err| return x.committedFault(err);
            if (reset) |q| _ = try done(x.structured_control.?, q, .{});
        }
        if (start.* >= ids.len) return error.PromptStepPast;
        const last = try x.chunk(f, ids, start.*);
        start.* += e.prefill_rows;
        if (start.* < ids.len) return null;
        return try x.firstToken(f, last);
    }

    /// decode.prefill's loop body for the chunk at `start`: begun MODE_CHUNK, staged, computed, its last row's logits
    /// and hidden row cloned, absorbed by the MTP head (`absorb`), committed; gives the logits' clone.
    pub fn chunk(x: *Attn, f: *Fwd, ids: []const u32, start: usize) !T {
        const e = x.e;
        const st = &e.st;
        const rows = ids[start..@min(start + e.prefill_rows, ids.len)];
        const permit: ?control.Permit = if (x.structured_control) |p| try p.before(.{ .kind = .prompt_chunk,
            .epoch = p.session.epoch, .peer_count = p.session.peer_count, .chunk_index = @intCast(start / e.prefill_rows),
            .position = start, .rows = @intCast(rows.len), .tokens = rows, .total_prompt_rows = ids.len }) else null;
        x.begin(f, rows.len, .chunk) catch |err| return x.committedFault(err);
        const R = cc.stage(f, st, &e.pbuf, rows) catch |err| return x.committedFault(err);
        const logits = (compute(e, f, &e.pbuf, R, .{ .nch = cc.chunksFor(st, R), .host_pos = st.pos, .chip = e.chip, .prompt = &e.prompt }) catch |err| return x.committedFault(err)).?;
        const last = cc.clone(f, &e.tmp, logits, &e.last) catch |err| return x.committedFault(err);
        _ = cc.clone(f, &e.tmp, cc.rows(e.pbuf.fnormed, R - 1, 1), &e.last_hidden) catch |err| return x.committedFault(err);
        x.launched();
        x.absorbChunk(f, &e.pbuf, ids, start, R) catch |err| return x.committedFault(err);
        cc.commit(f, st, &e.pbuf, R, R) catch |err| return x.committedFault(err);
        if (permit) |q| {
            settle(f, x) catch |err| return x.committedFault(err);
            x.finish(f) catch |err| return x.committedFault(err);
            _ = try done(x.structured_control.?, q, .{ .logits_valid = true });
        }
        return last;
    }

    /// qualification_21e21bc0's absorb of the R-row chunk at `start` (with `absorb`): its next tokens through the MTP head over its
    /// final-normed rows in `b` (the prompt Buffers, or qualification_0f991257's B view), then the MTP cache's length.
    pub fn absorbChunk(x: *Attn, f: *Fwd, b: *bufs.Buffers, ids: []const u32, start: usize, R: usize) !void {
        if (!x.absorb or start + 1 >= ids.len) return;
        const st = &x.e.st;
        const next = ids[start + 1 .. @min(start + R + 1, ids.len)];
        _ = try x.mtpStep(f, b, next, cc.rows(b.fnormed, 0, next.len), .chunk, &x.e.prompt);
        try cc.setMtpLen(f, st, st.mtp_len + next.len); // decode._absorb_rows
    }

    /// After the last chunk: the sync and the check (then the audit), and only then the first token's greedy draw.
    pub fn firstToken(x: *Attn, f: *Fwd, last: T) !u32 {
        settle(f, x) catch |err| return x.committedFault(err);
        x.finish(f) catch |err| return x.committedFault(err);
        if (x.defer_first) {
            if (x.structured_control) |p| {
                const permit = try p.before(.{ .kind = .group_action, .epoch = p.session.epoch,
                    .peer_count = p.session.peer_count, .action = .finish, .position = x.e.st.pos });
                _ = try done(p, permit, .{ .logits_valid = true });
            }
            x.deferred_logits = last;
            return 0; // structured-only deferral, never a selected/output token
        }
        var first: [1]u32 = .{0};
        if (x.e.sampling) |s| {
            if (s.temperature > 0) try x.e.sampleRows(f, last, x.e.st.pos, &first, null) else try sample.greedy(f, &x.e.smp, last, &first);
        } else try sample.greedy(f, &x.e.smp, last, &first);
        return first[0];
    }

    /// qualification_21e21bc0: one MTP head forward under AFD (cuda_afd_mtp.forward over `b`): the MTP slice begun (the halves' last
    /// position), its launches queued, its logits [1, V] valid only after `settled`. The caller advances mtp_len.
    pub fn mtpStep(x: *Attn, f: *Fwd, b: *bufs.Buffers, next: []const u32, hidden: T, mode: afd.Mode, p: ?*prompt.Prompt) !T {
        try x.beginOf(f, next.len, mode, true);
        const logits = if (mode == .eager and p == null and x.graphable(next.len)) try x.mtpGraphed(f, b, next, hidden) else try amtp.forward(f, x.e.w, &x.e.st, b, next, hidden, x.e.chip, p);
        x.launched();
        return logits;
    }

    /// The stream drained and the exchange checked (then the audit of the forward last launched): a draw may read.
    pub fn settled(x: *Attn, f: *Fwd) !void {
        try settle(f, x);
        try x.finish(f);
    }
};

/// forward.compute with logits, through the engine's recorder taps when one is set (B3's Engine.compute).
fn compute(e: *eng.Engine, f: *Fwd, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !?T {
    if (e.tap) |t| return dump.compute(f, e.w, &e.st, b, R, opt, true, t);
    return cc.compute(f, e.w, &e.st, b, R, opt, true);
}

/// qualification_f16009a8: a decode window's or an MTP step's compute as a graph body: its launches, and the provider's position (the one
/// host state they change: `restore` before them, `replayed` as they leave it); its logits' view.
const Body = struct {
    x: *Attn,
    f: *Fwd,
    b: *const bufs.Buffers,
    R: usize,
    opt: layer.Opt,
    mtp: bool,
    out: ?T = null,

    pub fn launch(y: *Body) !void {
        const e = y.x.e;
        y.out = if (y.mtp) try amtp.compute(y.f, e.w, &e.st, y.b, y.R, y.opt, true) else (try compute(e, y.f, y.b, y.R, y.opt)).?;
    }

    pub fn restore(y: *Body) void {
        y.x.a.pos = y.x.a.first;
        y.out = null;
    }

    pub fn replayed(y: *Body) void {
        y.x.a.pos = y.x.a.first + y.x.a.count;
        // cc.compute's logits of a decode window (from row 0), amtp.compute's of the last row
        y.out = if (y.mtp) T.rows2(y.b.logits, 0, 1, y.x.e.w.head.n(), y.b.logits.shape[1]) else layer.rowsOf(y.b.logits, y.R);
    }

    pub fn state(y: *const Body) struct { pos: usize, out: ?T } {
        return .{ .pos = y.x.a.pos, .out = y.out };
    }
};

/// The log's line for a failed exchange (PLAN §4 item 5): `afd: failed <why> peer <k> seq <n> gave_up <g> breach <b>`.
pub fn describe(x: xchg.Failure, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "afd: failed {s} peer {d} seq {d} gave_up {d} breach {d}", .{ std.enums.tagName(xchg.proto.Why, x.why) orelse "unknown", x.peer, x.seq, x.gave_up, x.breach }) catch "afd: failed";
}
