//! Row-invariant numerical stand-in for recorded MTP: each output depends on its staged token row.
//! NOT CUDA numerics; wrong logits row selection changes the returned token and fails identity tests.
const std = @import("std");
const glm = @import("glm_cuda");
const rec = @import("shared_verify_rec_test.zig");
pub const Picks = struct {
    r: *rec.Rec,
    draws: usize = 0,
    fail_at: ?usize = null,
    cancel_at: ?usize = null,
    flag: bool = false,
    shared: bool = true,
    pub fn token(id: u32) u32 {
        return (id *% 13 +% 7) % 127;
    }
    pub fn pick(p: *anyopaque, logits: glm.trace.T, _: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        const x: *Picks = @ptrCast(@alignCast(p));
        if (x.fail_at == x.draws) return error.DraftFault;
        if (x.cancel_at == x.draws) x.flag = true;
        const mb = &x.r.e.mbuf.?;
        const is_mtp = logits.ptr >= mb.logits.ptr and logits.ptr < mb.logits.ptr + mb.logits.bytes();
        var tokens: ?[]const glm.trace.Val = null;
        var k = x.r.trace.events.items.len;
        while (k > 0) {
            k -= 1;
            const event = x.r.trace.events.items[k];
            if (!std.mem.eql(u8, event.name, "host.write")) continue;
            if (is_mtp and event.vals[0].t.ptr != mb.ids_host.ptr) continue;
            tokens = event.vals[1].list;
            break;
        }
        const ids = tokens orelse return error.NoStagedRows;
        const row = if (is_mtp and x.r.users.bd != null and x.shared) @as(usize, @intCast(logits.ptr - mb.logits.ptr)) / (logits.stride[0] * logits.dtype.size()) else if (out.len > 1) 0 else ids.len - 1;
        if (row + out.len > ids.len) return error.BadLogitsRow;
        for (out, 0..) |*v, i| v.* = token(@intCast(ids[row + i].int));
        if (probs) |ps| @memset(ps, 1.0);
        x.draws += out.len;
    }
    pub fn cancelled(p: ?*anyopaque) bool {
        const x: *Picks = @ptrCast(@alignCast(p.?));
        return x.flag;
    }
};
