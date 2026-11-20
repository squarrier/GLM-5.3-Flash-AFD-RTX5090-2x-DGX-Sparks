// SPDX-License-Identifier: Apache-2.0
//! Private I3 original accounting delta for TensorFold v1.2.8.
//! GPT-6.1 Sol via OpenAI Codex, for Scott Quarrier.
//! Native loader/page/transport formula attribution is retained in PROVENANCE.md.
//! Pure metadata only. This module never probes, loads, allocates, pins or maps model memory.
const std = @import("std");
pub const Role = enum { attention, expert0, expert1 };
pub const WeightClass = enum { attention, dense, shared, router, embedding_head };
pub const DType = enum { bf16, f16, f32, i32, i16, u8 };
pub const Weight = struct {
    identity: u64,
    class: WeightClass,
    stored: DType,
    resident: DType,
    elements: u64,
};
pub const Exl3 = struct { experts: u64, hidden: u64, inter: u64, layers: u64 };
pub const WeightTotals = struct {
    attention: u64 = 0,
    dense: u64 = 0,
    shared: u64 = 0,
    router: u64 = 0,
    embedding_head: u64 = 0,
    routed: u64 = 0,
    total: u64 = 0,
};

pub fn weightBytes(role: Role, ws: []const Weight, ex: Exl3) !WeightTotals {
    if (ex.experts != 288 or ex.hidden != 4096 or ex.inter != 2048 or ex.layers == 0 or ex.layers > 43) return error.UnsupportedGeometry;
    if (role != .attention) {
        const width = ex.inter / 2;
        const trellis = try std.math.mul(u64, try std.math.mul(u64, ex.hidden, width), 3);
        const scales = try std.math.mul(u64, try std.math.add(u64, ex.hidden, width), 6);
        const per = try std.math.add(u64, try std.math.add(u64, trellis / 2, scales), 28);
        const n = try std.math.mul(u64, try std.math.mul(u64, per, ex.experts), ex.layers);
        return .{ .routed = n, .total = n };
    }
    var out: WeightTotals = .{};
    outer: for (ws, 0..) |w, i| {
        if (w.identity == 0) return error.BadWeightIdentity;
        for (ws[0..i]) |previous| {
            if (previous.identity == w.identity) {
                if (!std.meta.eql(previous, w)) return error.IdentityConflict;
                continue :outer;
            }
        }
        if (w.stored != w.resident and !(w.stored == .bf16 and w.resident == .f32)) return error.UnsupportedConversion;
        const width: u64 = switch (w.resident) { .bf16, .f16, .i16 => 2, .f32, .i32 => 4, .u8 => 1 };
        const n = try std.math.mul(u64, w.elements, width);
        switch (w.class) {
            .attention => out.attention = try std.math.add(u64, out.attention, n),
            .dense => out.dense = try std.math.add(u64, out.dense, n),
            .shared => out.shared = try std.math.add(u64, out.shared, n),
            .router => out.router = try std.math.add(u64, out.router, n),
            .embedding_head => out.embedding_head = try std.math.add(u64, out.embedding_head, n),
        }
        out.total = try std.math.add(u64, out.total, n);
    }
    return out;
}

fn add(a: u64, b: u64) !u64 { return std.math.add(u64, a, b); }
fn mul(a: u64, b: u64) !u64 { return std.math.mul(u64, a, b); }
fn aligned(n: u64, alignment: u64) !u64 { return try mul((try add(n, alignment - 1)) / alignment, alignment); }

pub const HostPool = struct { page_stride: u64, host_pool: u64, device_keys: u64, gather_arena_rows: u64, device_gather: u64, pinned_overflow: u64 };
/// Native state.pageBytes includes pooled keys even in the pinned host backing.
/// prepare_pages is hostPages(), NOT necessarily the smaller attached pool page count.
/// The native +1 cache slot is reserved independently of whether MTP execution is enabled.
pub fn hostPoolBytes(pages: u64, prepare_pages: u64, mla_layers: u64, latent: u64, index_dim: u64, cap: u64) !HostPool {
    if (pages == 0 or prepare_pages < pages or mla_layers == 0 or latent == 0 or index_dim == 0 or cap == 0) return error.BadShape;
    const layers = try add(mla_layers, 1);
    const row_bytes = try mul(latent, 2);
    const keys = try mul(512 / 4, try mul(index_dim, 2));
    const stride = try mul(layers, try add(try mul(512, row_bytes), keys));
    const arena_rows = (try mul(prepare_pages, stride)) / row_bytes;
    var gather = try mul(arena_rows, 8); // slot_of plus prev_of, both I32
    gather = try add(gather, try mul(cap, try add(4, row_bytes))); // list plus slots
    gather = try add(gather, 16 + 16 * 8192 * 4 + 16 * 4); // n, prev_rows, prev_n
    gather = try add(gather, try mul(16 * 8192, row_bytes)); // layer row cache
    return .{ .page_stride = stride, .host_pool = try mul(pages, stride), .device_keys = try mul(pages, try mul(layers, keys)),
        .gather_arena_rows = arena_rows, .device_gather = gather, .pinned_overflow = 16 };
}

pub const WireGeometry = struct { native_host_regions: u64, native_device_state: u64, request_per_pair: u64, reply_per_pair: u64, post_ceiling: u64 };
/// Payload and native region geometry ONLY, not a claim about I1's additional staging/verbs allocations.
/// Native mapped host regions alias a device address; they are not also device VRAM.
pub fn wireGeometry(role: Role, tile_rows: u64) !WireGeometry {
    if (tile_rows == 0 or tile_rows > 128) return error.BadShape;
    const pairs: u64 = if (role == .attention) 2 else 1;
    return .{ .native_host_regions = pairs * (8192 + 4 * 1048576), .native_device_state = pairs * 16,
        .request_per_pair = try add(try aligned(try mul(tile_rows, 4096 * 2), 256), try mul(tile_rows, 8 * 4)),
        .reply_per_pair = try mul(tile_rows, 8 * 4096 * 4), .post_ceiling = 1048576 };
}

pub const ledger_capacity = 2048; // bounded metadata; Phase B must check its full backing census
pub const Owner = enum { weights, host_pages, pooled_keys, gather, sequence_state, snapshots, call_planes, transport, workspace, load_staging, engram_external, driver_external };
pub const Charge = struct {
    device: u64 = 0,
    host: u64 = 0,
    pinned: u64 = 0, // subset of host, NEVER added again to physical RAM
    registered: u64 = 0, // subset of host; overlaps pinned, not an extra physical allocation

    pub fn plus(a: Charge, b: Charge) !Charge {
        return .{ .device = try add(a.device, b.device), .host = try add(a.host, b.host),
            .pinned = try add(a.pinned, b.pinned), .registered = try add(a.registered, b.registered) };
    }
    fn maximum(a: Charge, b: Charge) Charge {
        return .{ .device = @max(a.device, b.device), .host = @max(a.host, b.host),
            .pinned = @max(a.pinned, b.pinned), .registered = @max(a.registered, b.registered) };
    }
};
pub const Allocation = struct {
    identity: u64,
    owner: Owner,
    charge: ?Charge, // unknown is not zero and cannot pass complete accounting
    stages: u3 = 7, // bit0 load, bit1 served, bit2 bounded close/quarantine
    preexisting: bool = false, // already present in the supplied free-memory observation
};
pub const Summary = struct {
    phases: [3]Charge = @splat(.{}),
    incremental: [3]Charge = @splat(.{}),
    peak: Charge = .{},
    incremental_peak: Charge = .{},
    incremental_unified_peak: u64 = 0,
};
/// Engine-thread metadata, no allocator and no global state. Identities name physical backing, not views.
pub const Plan = struct {
    items: [ledger_capacity]Allocation = undefined,
    len: usize = 0,

    pub fn put(p: *Plan, a: Allocation) !void {
        if (a.identity == 0 or a.stages == 0) return error.BadAllocation;
        if (a.charge) |c| if ((c.pinned != 0 and c.pinned != c.host) or (c.registered != 0 and c.registered != c.host)) return error.BadAllocation;
        for (p.items[0..p.len]) |old| if (old.identity == a.identity) {
            if (!std.meta.eql(old, a)) return error.IdentityConflict;
            return;
        };
        if (p.len == p.items.len) return error.PlanFull;
        p.items[p.len] = a;
        p.len += 1;
    }
    pub fn summarize(p: *const Plan, required: []const Owner) !Summary {
        for (required) |owner| {
            var found = false;
            for (p.items[0..p.len]) |a| if (a.owner == owner) { found = true; };
            if (!found) return error.MissingOwner;
        }
        var s: Summary = .{};
        for (p.items[0..p.len]) |a| {
            const c = a.charge orelse return error.UnknownAllocation;
            for (0..3) |stage| {
                if (a.stages & (@as(u3, 1) << @intCast(stage)) == 0) continue;
                s.phases[stage] = try s.phases[stage].plus(c);
                if (!a.preexisting) s.incremental[stage] = try s.incremental[stage].plus(c);
            }
        }
        for (s.phases, s.incremental) |total, inc| {
            s.peak = s.peak.maximum(total);
            s.incremental_peak = s.incremental_peak.maximum(inc);
            s.incremental_unified_peak = @max(s.incremental_unified_peak, try add(inc.device, inc.host));
        }
        return s;
    }
};

/// The native snapshot pool floors budget/page once. Leases only borrow its backing.
pub fn snapshotPoolBytes(budget: u64, page: u64, host: bool) !Charge {
    if (page == 0) return error.BadShape;
    const count = budget / page;
    if (count > std.math.maxInt(u32)) return error.SnapshotBudget;
    const n = try mul(count, page);
    return if (host) .{ .host = n, .pinned = n, .registered = n } else .{ .device = n };
}

pub fn arenaBytes(sizes: []const u64) !u64 {
    var total: u64 = 0;
    for (sizes) |n| total = try add(total, try aligned(n, 256));
    return @max(total, 256);
}
pub const StateShape = struct { kda_layers: u64, mla_layers: u64, heads: u64, head_dim: u64, row_width: u64, capacity: u64, index_dim: u64, hidden: u64, key_dim: u64, value_dim: u64 };
/// Exact cuda_state.stateSizes(table!=null) arithmetic. Other native buffer/sampler/heap
/// owners must be supplied separately; this is not the whole engine's per-user price.
pub fn pagedStateBytes(c: StateShape) !u64 {
    if (c.capacity == 0 or c.head_dim == 0 or c.heads == 0 or c.row_width == 0 or c.index_dim == 0 or c.hidden == 0) return error.BadShape;
    const n = try mul(c.heads, c.head_dim);
    const window_n = try mul(try mul(c.kda_layers, 32), n);
    const sizes = [_]u64{
        4, try mul(try mul(c.kda_layers, n), 3 * 3 * 2),
        try mul(try mul(try mul(try mul(c.kda_layers, c.heads), c.key_dim), c.value_dim), 2 * 4),
        try mul(try mul(try mul(c.kda_layers, 32), c.row_width), 2),
        try mul(window_n, 4), try mul(window_n, 2), try mul(window_n, 4),
        try mul(try mul(try mul(c.kda_layers, 32), c.heads), 4),
        4, try mul(c.hidden, 2), try mul((try add(c.capacity, 511)) / 512, 4),
    };
    const rings = try mul(try mul(@min(c.capacity, 4096), c.index_dim), 2);
    // Each of MLA+1 caches has two rings and zero latent/pk fields when paged.
    const ring_total = try mul(try mul(try aligned(rings, 256), 2), try add(c.mla_layers, 1));
    return try add(try arenaBytes(&sizes), ring_total);
}

pub const Observation = struct { device_free: u64, host_available: u64, memlock_free: u64, observed_ns: u64, valid_until_ns: u64, epoch: u64 };
pub const Policy = struct { device_floor: u64, host_floor: u64, unified: bool };
pub const Reservation = struct { epoch: u64, sequence: u64, summary: Summary };
/// One engine-thread bank per observation. Do not copy/renew it or create parallel
/// banks over the same physical budget. Cleanup never manufactures a fresh observation.
pub const Budget = struct {
    observation: Observation,
    policy: Policy,
    device_left: u64,
    host_left: u64,
    memlock_left: u64,
    unified_left: u64,
    sequence: u64 = 0,

    pub fn init(o: Observation, policy: Policy, now: u64) !Budget {
        if (o.epoch == 0 or o.valid_until_ns <= o.observed_ns or now < o.observed_ns or now > o.valid_until_ns) return error.StaleObservation;
        if (o.device_free < policy.device_floor) return error.DeviceFloor;
        if (o.host_available < policy.host_floor) return error.HostFloor;
        return .{ .observation = o, .policy = policy, .device_left = o.device_free - policy.device_floor,
            .host_left = o.host_available - policy.host_floor, .memlock_left = o.memlock_free,
            .unified_left = @min(o.device_free - policy.device_floor, o.host_available - policy.host_floor) };
    }
    pub fn reserve(b: *Budget, p: *const Plan, required: []const Owner, now: u64) !Reservation {
        if (now < b.observation.observed_ns or now > b.observation.valid_until_ns) return error.StaleObservation;
        const s = try p.summarize(required);
        if (s.incremental_peak.device > b.device_left) return error.DeviceFloor;
        if (s.incremental_peak.host > b.host_left) return error.HostFloor;
        if (b.policy.unified and s.incremental_unified_peak > b.unified_left) return error.UnifiedFloor;
        var locks: [3]u64 = @splat(0);
        for (p.items[0..p.len]) |a| {
            if (a.preexisting) continue;
            const c = a.charge orelse return error.UnknownAllocation;
            // Whole-backing pinned/registered descriptors; the same backing is locked once.
            for (0..3) |stage| if (a.stages & (@as(u3, 1) << @intCast(stage)) != 0) {
                locks[stage] = try add(locks[stage], @max(c.pinned, c.registered));
            };
        }
        const locked = @max(locks[0], @max(locks[1], locks[2]));
        if (locked > b.memlock_left) return error.MemlockFloor;
        const next = try add(b.sequence, 1);
        b.device_left -= s.incremental_peak.device;
        b.host_left -= s.incremental_peak.host;
        b.memlock_left -= locked;
        if (b.policy.unified) b.unified_left -= s.incremental_unified_peak;
        b.sequence = next;
        return .{ .epoch = b.observation.epoch, .sequence = next, .summary = s };
    }
};

pub const CloseEvidence = struct { incarnation: u128, quiet: bool, retained_ids: []const u64, released_ids: []const u64 = &.{} };
pub const ReleaseEvidence = struct { incarnation: u128, identity: u64, quiet: bool, owned_free_completed: bool };
pub const AllocationState = enum { planned, live, released, retained };
pub const Status = struct {
    role: Role,
    incarnation: u128,
    sticky_failure: bool,
    cancelled: bool,
    clean: bool,
    live: Charge = .{},
    retained: Charge = .{},
    external: Charge = .{},
    not_attempted: usize = 0,
    proof: []const u8 = "owned_allocation_balance_not_physical_leak_proof",
};
/// Records callbacks from actual owners; it does NOT allocate/free or infer completion from free RAM.
pub const Accounting = struct {
    role: Role,
    plan: Plan,
    incarnation: u128,
    states: [ledger_capacity]AllocationState = @splat(.planned),
    sticky_failure: bool = false,
    closed: bool = false,
    cancelled: bool = false,

    pub fn init(role: Role, plan: Plan, required: []const Owner, incarnation: u128) !Accounting {
        _ = try plan.summarize(required);
        if (incarnation == 0) return error.IdentityMismatch;
        return .{ .role = role, .plan = plan, .incarnation = incarnation };
    }
    pub fn acquired(a: *Accounting, identity: u64) !void {
        if (a.closed) return error.Closed;
        if (a.sticky_failure) return error.StickyFailure;
        for (a.plan.items[0..a.plan.len], 0..) |item, i| if (item.identity == identity) {
            if (item.preexisting) return error.ExternalOwner;
            if (a.states[i] != .planned) return error.AlreadyAcquired;
            a.states[i] = .live;
            return;
        };
        return error.UnknownAllocation;
    }
    pub fn fail(a: *Accounting) void { a.sticky_failure = true; }
    pub fn released(a: *Accounting, proof: ReleaseEvidence) !void {
        if (a.closed) return error.Closed;
        if (proof.incarnation != a.incarnation) return error.IdentityMismatch;
        if (!proof.quiet or !proof.owned_free_completed) return error.NoReleaseProof;
        for (a.plan.items[0..a.plan.len], 0..) |item, i| if (item.identity == proof.identity) {
            if (item.preexisting) return error.ExternalOwner;
            if (a.states[i] != .live) return error.AlreadyReleased;
            a.states[i] = .released;
            return;
        };
        return error.UnknownAllocation;
    }
    pub fn close(a: *Accounting, proof: CloseEvidence) !void {
        if (a.closed) return error.Closed;
        if (proof.incarnation != a.incarnation) return error.IdentityMismatch;
        if (!proof.quiet and proof.released_ids.len != 0) return error.InvalidCloseReceipt;
        var disposition: [ledger_capacity]u2 = @splat(0);
        // Validate the whole typed receipt before mutating any allocation state.
        for ([_][]const u64{ proof.retained_ids, proof.released_ids }, 0..) |ids, which| {
            for (ids) |id| {
                var found = false;
                for (a.plan.items[0..a.plan.len], 0..) |item, i| if (item.identity == id) {
                    if (item.preexisting or a.states[i] != .live or disposition[i] != 0) return error.InvalidCloseReceipt;
                    disposition[i] = @intCast(which + 1);
                    found = true;
                };
                if (!found) return error.InvalidCloseReceipt;
            }
        }
        for (a.states[0..a.plan.len], 0..) |*state, i| {
            if (state.* != .live) continue;
            state.* = if (proof.quiet and disposition[i] == 2) .released else .retained;
            if (state.* == .retained) a.sticky_failure = true;
        }
        if (!proof.quiet) a.sticky_failure = true;
        a.closed = true;
    }
    pub fn status(a: *const Accounting) !Status {
        var s: Status = .{ .role = a.role, .incarnation = a.incarnation, .sticky_failure = a.sticky_failure, .clean = a.closed, .cancelled = a.cancelled };
        for (a.plan.items[0..a.plan.len], a.states[0..a.plan.len]) |item, state| {
            const c = item.charge orelse return error.UnknownAllocation;
            if (item.preexisting) { s.external = try s.external.plus(c); continue; }
            switch (state) {
                .live => { s.live = try s.live.plus(c); s.clean = false; },
                .retained => { s.retained = try s.retained.plus(c); s.clean = false; },
                .planned => s.not_attempted += 1,
                .released => {},
            }
        }
        return s;
    }
    pub fn assertReleased(a: *const Accounting) !void {
        for (a.states[0..a.plan.len]) |state| {
            if (state == .retained) return error.RetainedAllocations;
            if (state == .live) return error.LiveAllocations;
        }
        if (!a.closed) return error.NotClosed;
    }
    pub fn cancel(a: *Accounting, proof: CloseEvidence) !void {
        try a.close(proof);
        a.cancelled = true;
    }
};

pub const Capacity = struct { logical_per_request: u64, served_per_request: u64, allocated_shared_rows: u64, resident_users: u32, qualified_per_request: ?u64 };
pub const GatherCounters = struct { calls: u64, gathered_rows: u64, cache_hits: u64, overflows: u64 };
pub const TransportCounters = struct { posted: u64, retired: u64, failures: u64 };
pub const Report = struct {
    schema: u32 = 1,
    scope: []const u8 = "source_bound_accounting_not_model_qualification",
    source_binding: [32]u8,
    status: Status,
    plan: Summary,
    capacity: Capacity,
    gather: ?GatherCounters,
    transport: ?TransportCounters,
};
/// The caller must bind the actual composed native source; a nonzero synthetic hash
/// in a CPU fixture remains a fixture, never recovered live evidence.
pub fn report(a: *const Accounting, cap: Capacity, gather: ?GatherCounters, transport: ?TransportCounters, source_binding: [32]u8) !Report {
    if (std.mem.allEqual(u8, &source_binding, 0)) return error.SourceUnbound;
    if (cap.logical_per_request == 0 or cap.served_per_request == 0 or cap.served_per_request > cap.logical_per_request or cap.resident_users == 0) return error.BadCapacity;
    if (cap.qualified_per_request) |qualified| if (qualified > cap.served_per_request) return error.BadCapacity;
    return .{ .source_binding = source_binding, .status = try a.status(), .plan = try a.plan.summarize(&.{}),
        .capacity = cap, .gather = gather, .transport = transport };
}

pub const Scratch = struct { halves: u64, epilogue: u64, routed_weights: u64, shared_weights: u64, total: u64 };
/// D1: per-Buffers durable planes, not transport tiles; native arenas round each field to 256.
pub fn scratchBytes(rows: u64, twin: bool) !Scratch {
    if (rows == 0) return error.BadShape;
    const copies: u64 = if (twin) 2 else 1;
    const row = try mul(rows, 4096 * 4);
    const halves = try mul(try mul(row, 2), copies);
    const epilogue = try mul(try mul(row, 3), copies);
    // The actual native hybrid arena is packed once; these are views, not separately aligned allocations.
    const rw = try mul(try mul(rows, 8 * 4), copies);
    const sw = try mul(try mul(rows, 4), copies);
    return .{ .halves = halves, .epilogue = epilogue, .routed_weights = rw, .shared_weights = sw,
        .total = try add(try add(halves, epilogue), try add(rw, sw)) };
}
