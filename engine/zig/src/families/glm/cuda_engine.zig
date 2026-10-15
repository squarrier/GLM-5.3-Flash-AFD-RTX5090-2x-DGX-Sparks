//! python-0.6's Engine (glm5_next/cuda/decode.py, ed78d6f) for a TP rank: the weights, one sequence's State, Buffers
//! for decode windows (`buf`), prompt chunks (`pbuf`) and the MTP head (`mbuf`), and the sampler. `prefill` commits a
//! prompt in chunks of prefill_rows (each chunk's next tokens absorbed into the MTP cache) and samples its first
//! token; cuda_decode.zig's loops then step (serial, MTP drafts verified: mtp_decode, auto_decode's MTP arm), here as
//! methods. Greedy, or with `sampling` at top_k 0 python's nucleus (qualification_45c10bcf: cuda_nucleus.zig, Mia's union cover).
//! Every call goes through *Fwd: a CPU test records python's launches, a GPU runs them (CUDA graphs: qualification_qualification_44ba554e.8).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const cc = @import("cuda_compute.zig");
const sample = @import("cuda_sample.zig");
const nucleus = @import("cuda_nucleus.zig");
const mtp = @import("cuda_mtp.zig");
const prompt = @import("cuda_prompt.zig");
const dump = @import("cuda_dump.zig");
const decode = @import("cuda_decode.zig");
const Tmp = @import("cuda_tmp.zig").Tmp;
const Fwd = @import("cuda_fwd.zig").Fwd;
const control = @import("lanes").structured_control;

const T = trace.T;
const Buf = views.Buf;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// decode.PREFILL_ROWS: rows of a prompt chunk
pub const prefill_rows = 2048;

/// The Engine with the MTP layer's blocks from `B` (mtp.LaneA; a test's markers).
pub fn EngineOf(comptime B: type) type {
    return struct {
        const Self = @This();

        m: *const md.Model,
        w: *const views.Weights,
        buf: Buffers,
        pbuf: Buffers,
        mbuf: ?Buffers,
        st: State,
        smp: sample.Sampler,
        tmp: Tmp = .{},
        prompt: prompt.Prompt,
        /// prefill's `last` (the last chunk's logits row) and `e.last_hidden` (its final-normed row), python's clones
        last: Buf,
        last_hidden: Buf,
        /// auto_decode's m_rows: committed rows the MTP head has not absorbed yet (decode.backlog_rows of them)
        backlog: Buf,
        prefill_rows: usize,
        chip: qmm.Chip = .{},
        /// qualification_6607f886's recorder on every prefill chunk and decode window (tf-glm-forward dump and teacher)
        tap: ?dump.Tap = null,
        /// The request's sampling (null, or temperature 0: greedy); top_k 0 draws the nucleus (useSampling first).
        sampling: ?nucleus.Sampling = null,
        nuc: ?nucleus.Nucleus = null,
        /// sampling_tests: the AFD serve switch enables the top-k cut; other lanes keep their admission rule.
        sampled_top_k: bool = false,
        /// The request's nucleus calls by the gathers each took (prefill starts it afresh).
        stats: nucleus.Stats = .{},
        structured_control: ?control.Endpoint = null,
        structured_logits_valid: bool = false,

        /// Engine(w, capacity=, max_rows=, prefill_rows=, long_context=), no graphs; addresses by `walk` or Device.attach.
        pub fn create(gpa: std.mem.Allocator, m: *const md.Model, w: *const views.Weights, capacity: usize, rows: usize, chunk: usize, long_context: bool) !*Self {
            const e = try gpa.create(Self);
            errdefer gpa.destroy(e);
            e.* = .{
                .m = m,
                .w = w,
                .buf = try Buffers.init(m, rows, capacity, false),
                .pbuf = try Buffers.init(m, chunk, capacity, true),
                .mbuf = if (w.mtp != null) try Buffers.init(m, rows, capacity, false) else null,
                .st = try State.init(gpa, m, capacity, rows, long_context),
                .smp = .{ .s = sample.Scratch.init(m, rows) },
                .prompt = undefined,
                .last = Buf.of(.bf16, &.{ 1, m.vl() }),
                .last_hidden = Buf.of(.bf16, &.{ 1, m.hidden }),
                .backlog = Buf.of(.bf16, &.{ decode.backlog_rows, m.hidden }),
                .prefill_rows = chunk,
            };
            e.prompt = prompt.Prompt.init(m, &e.tmp, chunk);
            e.prompt.dsa = prompt.dsaScratch(m, &e.pbuf, capacity, long_context); // long contexts: chunks past lat_s
            e.smp.tmp = &e.tmp;
            return e;
        }

        pub fn destroy(e: *Self, gpa: std.mem.Allocator) void {
            if (e.nuc) |*n| n.deinit(gpa);
            e.st.deinit(gpa);
            gpa.destroy(e);
        }

        /// The nucleus sampler's rows (device, pinned and host), before Device.attach: sampled requests need them.
        pub fn useSampling(e: *Self, gpa: std.mem.Allocator) !void {
            if (e.nuc == null) e.nuc = try nucleus.Nucleus.init(gpa, e.m, e.buf.rows);
        }

        /// python's buffers by path (b, pbuf, mbuf, st), for a Space or an arena.
        pub fn walk(e: *Self, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
            try views.walk(&e.buf, views.Path.init("b"), ctx, visit);
            try views.walk(&e.pbuf, views.Path.init("pbuf"), ctx, visit);
            if (e.mbuf) |*x| try views.walk(x, views.Path.init("mbuf"), ctx, visit);
            try views.walk(&e.st, views.Path.init("st"), ctx, visit);
        }

        /// The engine's own rows for python's temporaries (the sampler's scratch, a cut layer's rows, prefill's clones,
        /// auto_decode's backlog, the nucleus rows when sampling is used; last the prompt chunks' DSA Scratch when
        /// reserved, so every other buffer keeps its place).
        pub fn walkTmp(e: *Self, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
            try views.walk(&e.smp.s, views.Path.init("tmp"), ctx, visit);
            try visit(ctx, "tmp.first", &e.prompt.first);
            try visit(ctx, "tmp.cat", &e.prompt.cat);
            try visit(ctx, "tmp.last", &e.last);
            try visit(ctx, "tmp.last_hidden", &e.last_hidden);
            try visit(ctx, "tmp.backlog", &e.backlog);
            if (e.nuc) |*n| try views.walk(&n.s, views.Path.init("tmp.nucleus"), ctx, visit);
            if (comptime prompt.DsaScratch != void) try views.walk(&e.prompt.dsa, views.Path.init("tmp.dsa"), ctx, visit);
        }

        /// decode.sample_rows on `logits` [R, V/world] at positions pos0 .. pos0 + R - 1: greedy without a sampling or at
        /// temperature 0, nucleus_rows at top_k 0 (python's top_k path is not ported). With `probs`, each draw's
        /// probability (a draft's confidence): greedy, decode._probability at temperature 1 under the request's top_p
        /// and top_k; nucleus, the token's share of the mass (python's nucleus_rows probs).
        pub fn sampleRows(e: *Self, f: *Fwd, logits: T, pos0: usize, out: []u32, probs: ?[]f64) !void {
            const s = e.sampling orelse return sample.greedyTop(f, &e.smp, logits, out, probs);
            if (s.temperature <= 0.0) return sample.greedyCut(f, &e.smp, logits, out, probs, .{ .top_p = s.top_p, .top_k = s.top_k });
            if (s.top_k != 0 and !e.sampled_top_k) return error.TopKNotPorted;
            const n = if (e.nuc) |*x| x else return error.NoNucleus;
            var pos: [nucleus.max_rows]u64 = undefined;
            const R = logits.shape[0];
            if (R > pos.len) return error.SampleShape;
            for (pos[0..R], 0..) |*q, i| q.* = pos0 + i;
            try nucleus.nucleusRows(f, &e.smp, n, logits, pos[0..R], s, out, probs, &e.stats);
        }

        /// structured_users: sampleRows' nucleus draw of one row [1, V] at `pos0` among `mask`'s tokens (TF_GLM_SAMPLING=1 with
        /// TF_AFD_GRAMMAR_MASK=1): the masked adapter's grammar_mask on the widened row before the temperature, the
        /// maxima, the masses and the cuts, then the token checked eligible (cuda_mask_dev.Dev.sampled). No mask, or
        /// every token allowed: nucleus_rows itself.
        pub fn sampleMasked(e: *Self, f: *Fwd, logits: T, pos0: usize, out: []u32, mask: []const u32) !void {
            const s = e.sampling orelse return error.NotNucleus;
            if (!(s.temperature > 0.0)) return error.NotNucleus;
            if (s.top_k != 0 and !e.sampled_top_k) return error.TopKNotPorted;
            const n = if (e.nuc) |*x| x else return error.NoNucleus;
            const d = e.smp.masked_dev orelse return error.IncompleteMaskedAdapter;
            try d.sampled(f, &e.smp, n, logits, pos0, s, out, &e.stats, mask);
        }

        /// The bytes Device.attach reserves (each buffer at a 256-byte boundary), and the prompt Buffers' and the DSA
        /// Scratch's among them (the log's line beside G-mem).
        pub fn deviceBytes(e: *Self) !DeviceBytes {
            var all: Sizer = .{};
            try e.walk(&all, Sizer.visit);
            try e.walkTmp(&all, Sizer.visit);
            var pb: Sizer = .{};
            try views.walk(&e.pbuf, views.Path.init("pbuf"), &pb, Sizer.visit);
            var x: Sizer = .{};
            if (comptime prompt.DsaScratch != void) try views.walk(&e.prompt.dsa, views.Path.init("tmp.dsa"), &x, Sizer.visit);
            return .{ .total = all.used, .pbuf = pb.used, .dsa = x.used };
        }

        // -- prompts ----------------------------------------------------------------------------------------------

        /// mtp.mtp_forward over B's blocks: stage, then compute at the MTP cache's length (the last row's logits); `p`, a
        /// prompt chunk's context on prefill's absorb (its DSA Scratch).
        fn mtpForward(f: *Fwd, w: *const views.Weights, st: *const State, b: *Buffers, next: []const u32, hidden: T, chip: qmm.Chip, p: ?*prompt.Prompt) !T {
            const n = try mtp.stage(f, st, b, next, hidden);
            const nch = (st.mtp_len + n + cc.chunk - 1) / cc.chunk;
            return mtp.computeWith(B, f, w, st, b, n, .{ .nch = nch, .host_pos = st.mtp_len, .chip = chip, .prompt = p }, true);
        }

        /// decode.prefill (no snapshot or drafter): each chunk staged, computed, absorbed by the MTP head, committed.
        pub fn prefill(e: *Self, f: *Fwd, ids: []const u32, use_mtp: bool) !u32 {
            if (ids.len == 0) return error.EmptyPrompt;
            const st = &e.st;
            e.stats = .{};
            try cc.reset(f, st);
            const with_mtp = use_mtp and e.w.mtp != null;
            var last: ?T = null;
            var start: usize = 0;
            while (start < ids.len) : (start += e.prefill_rows) {
                const chunk = ids[start..@min(start + e.prefill_rows, ids.len)];
                const R = try cc.stage(f, st, &e.pbuf, chunk);
                const logits = (try e.compute(f, &e.pbuf, R, .{ .nch = cc.chunksFor(st, R), .host_pos = st.pos, .chip = e.chip, .prompt = &e.prompt })).?;
                last = try cc.clone(f, &e.tmp, logits, &e.last);
                _ = try cc.clone(f, &e.tmp, cc.rows(e.pbuf.fnormed, R - 1, 1), &e.last_hidden);
                if (with_mtp and start + 1 < ids.len) {
                    const next = ids[start + 1 .. @min(start + R + 1, ids.len)];
                    _ = try mtpForward(f, e.w, st, &e.pbuf, next, cc.rows(e.pbuf.fnormed, 0, next.len), e.chip, &e.prompt);
                    try cc.setMtpLen(f, st, st.mtp_len + next.len); // decode._absorb_rows
                }
                try cc.commit(f, st, &e.pbuf, R, R);
            }
            var first: [1]u32 = undefined;
            try e.sampleRows(f, last.?, ids.len, &first, null);
            return first[0];
        }

        /// Structured-only deferred prompt; ordinary prefill above remains byte-identical.
        pub fn prefillToLogits(e: *Self, f: *Fwd, ids: []const u32) !void {
            try sample.maskedReady(f, &e.smp);
            if (ids.len == 0) return error.EmptyPrompt;
            const endpoint = e.structured_control;
            e.structured_logits_valid = false;
            errdefer e.structured_logits_valid = false;
            if (endpoint) |p| {
                if (!@import("builtin").is_test or f.gpu != null or f.rec == null) return error.IncompleteMaskedAdapter;
                try p.open();
                const reset = try p.before(.{ .kind = .reset, .epoch = 1, .peer_count = 2, .total_prompt_rows = ids.len });
                cc.reset(f, &e.st) catch |err| return e.committedFault(endpoint, err);
                try structuredDone(p, reset, false);
            } else try cc.reset(f, &e.st);
            const st = &e.st;
            e.stats = .{};
            var start: usize = 0;
            while (start < ids.len) : (start += e.prefill_rows) {
                const chunk = ids[start..@min(start + e.prefill_rows, ids.len)];
                const permit: ?control.Permit = if (endpoint) |p| try p.before(.{ .kind = .prompt_chunk, .epoch = 1, .peer_count = 2,
                    .chunk_index = @intCast(start / e.prefill_rows), .position = start, .rows = @intCast(chunk.len),
                    .tokens = chunk, .total_prompt_rows = ids.len }) else null;
                const R = cc.stage(f, st, &e.pbuf, chunk) catch |err| return e.committedFault(endpoint, err);
                const logits = (e.compute(f, &e.pbuf, R, .{ .nch = cc.chunksFor(st, R), .host_pos = st.pos, .chip = e.chip, .prompt = &e.prompt }) catch |err| return e.committedFault(endpoint, err)).?;
                _ = cc.clone(f, &e.tmp, logits, &e.last) catch |err| return e.committedFault(endpoint, err);
                _ = cc.clone(f, &e.tmp, cc.rows(e.pbuf.fnormed, R - 1, 1), &e.last_hidden) catch |err| return e.committedFault(endpoint, err);
                cc.commit(f, st, &e.pbuf, R, R) catch |err| return e.committedFault(endpoint, err);
                if (permit) |q| try structuredDone(endpoint.?, q, true);
            }
            if (endpoint) |p| {
                const finish = try p.before(.{ .kind = .group, .epoch = 1, .peer_count = 2, .position = ids.len, .total_prompt_rows = ids.len });
                try structuredDone(p, finish, true);
            }
            e.structured_logits_valid = true;
        }

        fn committedFault(e: *Self, endpoint: ?control.Endpoint, original: anyerror) anyerror {
            e.structured_logits_valid = false;
            if (endpoint) |p| { _ = p.quiesce(.poison) catch {}; return error.ControlPoison; }
            return original;
        }

        fn structuredDone(endpoint: control.Endpoint, permit: control.Permit, valid: bool) !void {
            const result = try endpoint.after(permit, .{ .logits_valid = valid });
            switch (result.status) {
                .ok => {}, .cancelled => return error.Cancelled, .empty_mask => return error.EmptyMask,
                .request_error => return error.StructuredRequestError, .poison => return error.ControlPoison,
            }
        }

        pub fn pickFirst(e: *Self, f: *Fwd, mask: []const u32) !u32 {
            try sample.mask_cpu.shape(e.m.vocab, mask);
            if (e.structured_control) |p| {
                if (!e.structured_logits_valid) return error.NoDeferredFirst;
                e.smp.masked_cpu.?.permit = try p.before(selectionDescriptor(e, .first, &.{}, mask, .select));
            }
            defer if (e.smp.masked_cpu) |cpu| { cpu.permit = null; };
            var out: [1]u32 = undefined;
            try sample.greedyMasked(f, &e.smp, T.whole(e.last), &out, mask);
            return out[0];
        }

        fn selectionDescriptor(e: *Self, kind: control.Kind, tokens: []const u32, mask: []const u32, action: control.Action) control.Descriptor {
            return .{ .kind = kind, .epoch = 1, .peer_count = 2, .position = e.st.pos + @intFromBool(kind == .step),
                .rows = 1, .tokens = tokens, .mask = mask, .vocab_width = @intCast(e.m.vocab), .active = mask.len != 0, .action = action };
        }

        pub fn stepMasked(e: *Self, f: *Fwd, token: u32, mask: []const u32) !u32 {
            try sample.mask_cpu.shape(e.m.vocab, mask);
            try sample.maskedReady(f, &e.smp);
            const endpoint = e.structured_control;
            errdefer e.structured_logits_valid = false;
            const forward_permit: ?control.Permit = if (endpoint) |p| try p.before(selectionDescriptor(e, .step, &.{token}, mask, .none)) else null;
            const logits = e.forward(f, &.{token}) catch |err| return e.committedFault(endpoint, err);
            if (forward_permit) |q| {
                try structuredDone(endpoint.?, q, true);
                e.smp.masked_cpu.?.permit = try endpoint.?.before(selectionDescriptor(e, .step, &.{token}, mask, .select));
            }
            defer if (e.smp.masked_cpu) |cpu| { cpu.permit = null; };
            var out: [1]u32 = undefined;
            try sample.greedyMasked(f, &e.smp, logits, &out, mask);
            cc.commit(f, &e.st, &e.buf, 1, 1) catch |err| return e.committedFault(endpoint, err);
            return out[0];
        }

        // -- steps --------------------------------------------------------------------------------------------------

        /// Engine.forward (eager): a decode window's logits [R, V/world] in buf.logits.
        pub fn forward(e: *Self, f: *Fwd, tokens: []const u32) !T {
            const R = try cc.stage(f, &e.st, &e.buf, tokens);
            return (try e.compute(f, &e.buf, R, .{ .nch = cc.chunksFor(&e.st, R), .host_pos = e.st.pos, .chip = e.chip })).?;
        }

        /// forward.compute with logits, through the recorder's taps when one is set.
        fn compute(e: *Self, f: *Fwd, b: *const Buffers, R: usize, opt: @import("cuda_layer.zig").Opt) !?T {
            if (e.tap) |t| return dump.compute(f, e.w, &e.st, b, R, opt, true, t);
            return cc.compute(f, e.w, &e.st, b, R, opt, true);
        }

        /// Engine.mtp: the MTP head on (hidden, next token) rows: the last row's logits [1, V/world].
        pub fn mtpStep(e: *Self, f: *Fwd, next: []const u32, hidden: T) !T {
            const mb = if (e.mbuf) |*x| x else return error.NoMtp;
            return mtpForward(f, e.w, &e.st, mb, next, hidden, e.chip, null);
        }

        /// Engine.main_hidden(0:n): the main rows the MTP head reads after a forward.
        pub fn mainHidden(e: *const Self, n: usize) T {
            return cc.rows(e.buf.fnormed, 0, n);
        }

        // -- the decode loops (cuda_decode.zig: python's decode.py functions over the Engine) ---------------------------

        /// decode.draft with no confidence floor (tf-glm-forward's teacher draws): absorb, then `count` chained drafts.
        pub fn draft(e: *Self, f: *Fwd, hidden: T, next: []const u32, count: usize, out: []u32) ![]u32 {
            var steps: usize = 0;
            return decode.draft(e, f, hidden, next, e.st.pos + 1, count, .{}, out, null, &steps);
        }

        /// decode.serial_decode's tokens: from `pending` one a step into `out`, to `count` or an eos (stop_eos); its
        /// counters come with cuda_decode.serial.
        pub fn serialDecode(e: *Self, f: *Fwd, pending: u32, count: usize, out: []u32, stop_eos: bool) ![]u32 {
            return (try decode.serial(e, f, pending, count, out, stop_eos)).tokens;
        }

        /// decode.mtp_decode: verify [last, drafts] to the first mismatch, then draft again (the policy's depth).
        pub fn mtpDecode(e: *Self, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool) !Decoded {
            return decode.mtp(e, f, pending, count, out, policy, stop_eos, null);
        }

        /// drafter_choice.auto_decode with no DFlash2 drafter (engine.py's default policy, `auto`): every round MTP.
        pub fn autoDecode(e: *Self, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool) !Decoded {
            return decode.auto(e, f, pending, count, out, policy, stop_eos, null);
        }
    };
}

/// python's Engine: the MTP layer over lane A's blocks.
pub const Engine = EngineOf(mtp.LaneA);

pub const Decoded = decode.Decoded;
pub const DepthPolicy = decode.DepthPolicy;

/// An Engine's device bytes: the whole block Device.attach allocates, its prompt Buffers' and its DSA Scratch's.
pub const DeviceBytes = struct { total: usize, pbuf: usize, dsa: usize };

/// bufs.Arena's sizing pass without placing: every device buffer at a 256-byte boundary (pinned host rows out).
const Sizer = struct {
    used: usize = 0,

    fn visit(s: *Sizer, path: []const u8, b: *Buf) anyerror!void {
        _ = path;
        if (!b.host) s.used = std.mem.alignForward(usize, s.used + b.bytes(), 256);
    }
};

/// A rank's device memory for an Engine: one block for its buffers, pinned ids and `.cpu()` rows, `staged` events.
pub const Device = struct {
    block: cuda.DeviceBuffer,
    host: [5]?cuda.HostBuffer = @splat(null),
    events: [3]?cuda.Event = @splat(null),

    pub fn attach(e: *Engine, d: *const cuda.Driver) !Device {
        var a: bufs.Arena = .{};
        try e.walk(&a, bufs.Arena.visit);
        try e.walkTmp(&a, bufs.Arena.visit);
        var dv: Device = .{ .block = try cuda.DeviceBuffer.alloc(d, a.used) };
        errdefer dv.detach();
        a = .{ .base = dv.block.ptr };
        try e.walk(&a, bufs.Arena.visit);
        try e.walkTmp(&a, bufs.Arena.visit);
        var i: usize = 0;
        for ([_]*Buffers{ &e.buf, &e.pbuf }) |b| {
            try pin(d, &dv, &i, &b.ids_host);
            dv.events[i - 1] = try cuda.Event.init(d, false);
            b.staged = dv.events[i - 1];
        }
        if (e.mbuf) |*b| {
            try pin(d, &dv, &i, &b.ids_host);
            dv.events[i - 1] = try cuda.Event.init(d, false);
            b.staged = dv.events[i - 1];
        }
        try pin(d, &dv, &i, &e.smp.s.cpu);
        if (e.nuc) |*n| try pin(d, &dv, &i, &n.s.host);
        return dv;
    }

    fn pin(d: *const cuda.Driver, dv: *Device, i: *usize, b: *Buf) !void {
        if (!b.host) return error.NotHostRows;
        dv.host[i.*] = try cuda.HostBuffer.alloc(d, b.bytes());
        b.ptr = @intFromPtr(dv.host[i.*].?.bytes.ptr);
        i.* += 1;
    }

    pub fn detach(dv: *Device) void {
        for (&dv.events) |*x| if (x.*) |*v| v.deinit();
        for (&dv.host) |*x| if (x.*) |*v| v.free();
        dv.block.free();
    }
};
