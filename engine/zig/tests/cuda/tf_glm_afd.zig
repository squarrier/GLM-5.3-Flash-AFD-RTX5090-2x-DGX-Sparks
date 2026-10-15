//! tf-glm-afd (qualification_064ac7c1, Ash's #214): GLM-5.3-Flash's two AFD roles as one lab tool over native RoCE exchange. `expert` serves half H
//! of the routed experts (qualification_cec3c04c's afd_expert; --selftest SEED computes seeded requests with no network, for LZ2a);
//! `attn` is the attention node at world 1 (check-weights, first and teacher: qualification_9051d0c1's load and provider, qualification_0fa2bee9's engine;
//! afd_attn.zig). `check-flags` parses a role's flags on a CPU; `check-audit` compares TF_AFD_CHECK's records (the
//! attention's and both experts'). Each run ends with one line, `AFD OK ...` or `AFD FAILED <why> peer <k>
//! seq <n>`; exit 0 OK, 1 usage or setup, 2 an expert failed (the request failed), 3 a lost peer.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const xchg = cuda.xchg;
/// qualification_cec3c04c's expert command: its flags (afd_expert.usage), serve, --selftest and --check-weights.
const afd_expert = @import("afd_expert.zig");
/// qualification_0fa2bee9's attention node: first, teacher (--dry on a CPU), check-weights, and check-audit.
const attn = @import("afd_attn.zig");

const usage =
    \\usage: tf-glm-afd expert [serve] --half H --master IP:PORT --model DIR [flags]   half H (0 or 1) of the routed experts
    \\       tf-glm-afd expert selftest --selftest SEED --half H --model DIR --out FILE [--rows 1,4,8,2048] [--moe-layers 3-7]
    \\       tf-glm-afd expert|attn check-weights --model DIR [--half H] [--layers 8] [--ref weights.json] [--dry]
    \\       tf-glm-afd attn first|teacher --master IP:PORT --model DIR --prompt FILE[:NAME] [--teacher STEPS.json] [flags]
    \\       tf-glm-afd attn first|teacher --dry --model DIR --prompt FILE[:NAME] ...   a CPU: the verb recorded (config.json)
    \\       tf-glm-afd attn bench --master IP:PORT --model DIR --prompt FILE [--bench decode=p1024:512,prefill=p1900,...]
    \\       tf-glm-afd attn serve --master IP:PORT --model DIR [flags] -- [tensorfold serve flags: --host --port --name --context ...]
    \\       tf-glm-afd attn serve --dry --model DIR -- [--context N]   a CPU: the RTX 5090's forecast for the form at that window
    \\       tf-glm-afd check-flags expert|attn [VERB] [flags]   the flags as parsed (no GPU, no network)
    \\       tf-glm-afd check-audit ATTN.jsonl E0.jsonl E1.jsonl   TF_AFD_CHECK's window over the three sides' records
    \\flags: --layers N (backbone layers, a cut model) --dev mlx5_0[,TWIN] --gid 3 --max-rows 2048 --silent-s 10
    \\       --give-up-s 20 --no-watcher; attn: --out DIR --steps 48 --capacity 2560 --triton DIR --fatbin FILE --audit
    \\       --bench SPEC (qualification_e84ee1bf's timed test; profile=N splits N steps' MoE layers); TF_GLM_DENSE=q4:DIR, TF_GLM_F8=head
    \\       expert: --fault silent:N|exit:N (at request N: stop answering, or _exit 3; the lab's silent and dead cases)
    \\exit: 0 OK, 1 usage or setup, 2 an expert failed (the request failed), 3 a lost peer
    \\
;

pub const Role = enum { expert, attn };
pub const Verb = enum { serve, selftest, @"check-weights", first, teacher, bench };

pub const Flags = struct {
    role: Role = .expert,
    verb: Verb = .serve,
    half: ?u32 = null,
    master: ?std.Io.net.IpAddress = null,
    model: []const u8 = "",
    layers: ?u32 = null,
    devs: []const u8 = "mlx5_0",
    gid: c_int = 3,
    max_rows: usize = 2048,
    silent_s: f64 = 10,
    give_up_s: f64 = 20,
    watcher: bool = true,
    seed: ?u64 = null,
    rows: []const usize = &default_rows,
    moe_layers: ?[2]u32 = null,
    out: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    teacher: ?[]const u8 = null,
    steps: usize = 48,
    capacity: usize = 2560,
    triton: ?[]const u8 = null,
    fatbin: ?[]const u8 = null,
    ref: ?[]const u8 = null,
    dry: bool = false,
    audit: bool = false,
    fault: ?Fault = null,
    /// attn bench's spec (afd_attn_bench.zig; null: qualification_e84ee1bf's)
    bench: ?[]const u8 = null,
    /// attn serve's (qualification_4be15bbf): Ash's `tensorfold serve` flags after `--`, the model --model's (afd_attn_serve.zig)
    serve_args: []const []const u8 = &.{},
};

/// The process's environment (main's), for attn serve's server (its keys, its request log, TF_GLM_CLEAR_THINKING).
pub var environ: ?*const std.process.Environ.Map = null;

/// `--fault silent:N|exit:N` (an expert): at request N it stops answering, or _exits 3 (the lab's dead and silent cases).
pub const Fault = struct { what: enum { silent, exit }, at: u32 };

fn parseFault(text: []const u8) !Fault {
    const c = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadFault;
    const what = std.meta.stringToEnum(@FieldType(Fault, "what"), text[0..c]) orelse return error.BadFault;
    return .{ .what = what, .at = std.fmt.parseInt(u32, text[c + 1 ..], 10) catch return error.BadFault };
}

const default_rows = [_]usize{ 1, 4, 8, 2048 };

/// `IP:PORT` (an IPv6 address in brackets): the attention node's control address.
pub fn parseMaster(text: []const u8) !std.Io.net.IpAddress {
    const c = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadMaster;
    const host = std.mem.trim(u8, text[0..c], "[]");
    const port = std.fmt.parseInt(u16, text[c + 1 ..], 10) catch return error.BadMaster;
    return std.Io.net.IpAddress.parse(host, port) catch error.BadMaster;
}

fn parseRows(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| try list.append(gpa, std.fmt.parseInt(usize, t, 10) catch return error.BadRows);
    if (list.items.len == 0) return error.BadRows;
    return list.toOwnedSlice(gpa);
}

/// `A-B` (or `A`): MoE layers A..B inclusive.
fn parseSpan(text: []const u8) ![2]u32 {
    const d = std.mem.indexOfScalar(u8, text, '-');
    const a = std.fmt.parseInt(u32, text[0 .. d orelse text.len], 10) catch return error.BadSpan;
    const b = if (d) |i| std.fmt.parseInt(u32, text[i + 1 ..], 10) catch return error.BadSpan else a;
    if (b < a) return error.BadSpan;
    return .{ a, b };
}

/// A role's arguments after the role: an optional verb first (`expert` serves unless told, `attn` needs one), then flags.
pub fn parse(gpa: std.mem.Allocator, role: Role, argv: []const [:0]const u8) !Flags {
    var f: Flags = .{ .role = role, .verb = .serve };
    var args = argv;
    if (args.len > 0 and !std.mem.startsWith(u8, args[0], "--")) {
        f.verb = std.meta.stringToEnum(Verb, args[0]) orelse return error.BadVerb;
        args = args[1..];
    } else if (role == .attn) return error.MissingVerb;
    const ok_verb = switch (role) {
        .expert => f.verb == .serve or f.verb == .selftest or f.verb == .@"check-weights",
        .attn => f.verb == .@"check-weights" or f.verb == .first or f.verb == .teacher or f.verb == .bench or f.verb == .serve,
    };
    if (!ok_verb) return error.BadVerb;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--")) { // attn serve: the rest are Ash's serve flags (qualification_4be15bbf)
            const rest = try gpa.alloc([]const u8, args.len - i - 1);
            for (args[i + 1 ..], rest) |x, *y| y.* = x;
            f.serve_args = rest;
            break;
        }
        if (std.mem.eql(u8, k, "--no-watcher")) f.watcher = false else if (std.mem.eql(u8, k, "--dry")) f.dry = true else if (std.mem.eql(u8, k, "--audit")) f.audit = true else {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            const v = args[i];
            const int = std.fmt.parseInt;
            if (std.mem.eql(u8, k, "--half")) f.half = try int(u32, v, 10) else if (std.mem.eql(u8, k, "--master")) f.master = try parseMaster(v) else if (std.mem.eql(u8, k, "--model")) f.model = v else if (std.mem.eql(u8, k, "--layers")) f.layers = try int(u32, v, 10) else if (std.mem.eql(u8, k, "--dev")) f.devs = v else if (std.mem.eql(u8, k, "--gid")) f.gid = try int(c_int, v, 10) else if (std.mem.eql(u8, k, "--max-rows")) f.max_rows = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--silent-s")) f.silent_s = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--give-up-s")) f.give_up_s = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--selftest")) {
                f.seed = try int(u64, v, 10);
                f.verb = .selftest;
            } else if (std.mem.eql(u8, k, "--rows")) f.rows = try parseRows(gpa, v) else if (std.mem.eql(u8, k, "--moe-layers")) f.moe_layers = try parseSpan(v) else if (std.mem.eql(u8, k, "--out")) f.out = v else if (std.mem.eql(u8, k, "--prompt")) f.prompt = v else if (std.mem.eql(u8, k, "--teacher")) f.teacher = v else if (std.mem.eql(u8, k, "--steps")) f.steps = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--capacity")) f.capacity = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--triton")) f.triton = v else if (std.mem.eql(u8, k, "--fatbin")) f.fatbin = v else if (std.mem.eql(u8, k, "--ref")) f.ref = v else if (std.mem.eql(u8, k, "--fault")) f.fault = try parseFault(v) else if (std.mem.eql(u8, k, "--bench")) f.bench = v else return error.UnknownFlag;
        }
    }
    try check(f);
    return f;
}

/// Each verb's required flags and ranges.
fn check(f: Flags) !void {
    if (f.max_rows < 1 or f.max_rows > 4096 or f.silent_s <= 0 or f.give_up_s <= 0) return error.BadBound;
    if (f.model.len == 0) return error.NoModel;
    if (f.role == .expert and (f.half == null or f.half.? > 1)) return error.BadHalf;
    switch (f.verb) {
        .serve => if (f.master == null and !(f.role == .attn and f.dry)) return error.NoMaster, // attn serve --dry: the forecast
        .first, .teacher, .bench => if (f.master == null and !f.dry) return error.NoMaster, // --dry: no network
        .selftest => {
            if (f.seed == null) return error.NoSeed;
            if (f.out == null) return error.NoOut;
            for (f.rows) |r| if (r < 1 or r > f.max_rows) return error.BadRows;
        },
        .@"check-weights" => {},
    }
    if (f.role == .attn and f.verb != .@"check-weights" and f.verb != .serve and f.prompt == null) return error.NoPrompt;
    if (f.verb == .teacher and f.teacher == null) return error.NoTeacher;
    if (f.bench != null and f.verb != .bench) return error.BenchIsBenchs;
    if (f.serve_args.len > 0 and !(f.role == .attn and f.verb == .serve)) return error.ServeArgsAreServes;
    if (f.role == .attn and f.verb == .serve and f.audit) return error.ServeAudits; // the audit's records are the tool's verbs'
}

/// How a run ends; `code` is the exit code and `line` the summary the lab gates read.
pub const Outcome = union(enum) {
    ok: []const u8,
    usage: []const u8,
    expert_failed: Failure,
    lost: Failure,

    pub const Failure = struct { why: []const u8, peer: ?u32 = null, seq: u64 = 0 };

    pub fn code(o: Outcome) u8 {
        return switch (o) {
            .ok => 0,
            .usage => 1,
            .expert_failed => 2,
            .lost => cuda.watch.lost_exit_code,
        };
    }

    pub fn line(o: Outcome, buf: []u8) []const u8 {
        const x = switch (o) {
            .ok => |what| return std.fmt.bufPrint(buf, "AFD OK {s}", .{what}) catch "AFD OK",
            .usage => |why| Failure{ .why = why },
            .expert_failed, .lost => |x| x,
        };
        var peer: [12]u8 = undefined;
        const p = if (x.peer) |k| std.fmt.bufPrint(&peer, "{d}", .{k}) catch "?" else "-";
        return std.fmt.bufPrint(buf, "AFD FAILED {s} peer {s} seq {d}", .{ x.why, p, x.seq }) catch "AFD FAILED";
    }
};

/// The exchange's failure as the role sees it: on the attention node an expert failed (exit 2); on an expert, the
/// attention node is lost (exit 3). `why` is proto.Why's name.
pub fn fromFailure(role: Role, x: xchg.Failure) Outcome {
    const f: Outcome.Failure = .{ .why = std.enums.tagName(xchg.proto.Why, x.why) orelse "unknown", .peer = x.peer, .seq = x.seq };
    return if (role == .attn) .{ .expert_failed = f } else .{ .lost = f };
}

/// The flags as one line (check-flags' report, and every run's first line).
pub fn describe(f: Flags, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("{t} {t}", .{ f.role, f.verb }) catch {};
    if (f.half) |h| w.print(" half {d}", .{h}) catch {};
    if (f.master) |m| w.print(" master {f}", .{m}) catch {};
    w.print(" model {s} layers ", .{f.model}) catch {};
    if (f.layers) |n| w.print("{d}", .{n}) catch {} else w.print("all", .{}) catch {};
    w.print(" dev {s} gid {d} max-rows {d} silent {d}s give-up {d}s watcher {}", .{ f.devs, f.gid, f.max_rows, f.silent_s, f.give_up_s, f.watcher }) catch {};
    if (f.seed) |s| w.print(" seed {d} rows {any} out {s}", .{ s, f.rows, f.out orelse "-" }) catch {};
    if (f.moe_layers) |s| w.print(" moe-layers {d}-{d}", .{ s[0], s[1] }) catch {};
    if (f.prompt) |p| w.print(" prompt {s} steps {d} capacity {d} audit {}", .{ p, f.steps, f.capacity, f.audit }) catch {};
    if (f.ref) |r| w.print(" ref {s}", .{r}) catch {};
    if (f.fault) |x| w.print(" fault {t}:{d}", .{ x.what, x.at }) catch {};
    if (f.verb == .bench) w.print(" bench {s}", .{f.bench orelse "default"}) catch {};
    if (f.role == .attn and f.verb == .serve) {
        w.print(" --", .{}) catch {};
        for (f.serve_args) |x| w.print(" {s}", .{x}) catch {};
    }
    if (f.dry) w.print(" dry", .{}) catch {};
    return w.buffered();
}

/// One run: the role's body where this tree has it, else a refusal that names the lane it waits for.
fn run(gpa: std.mem.Allocator, io: std.Io, argv: []const [:0]const u8) Outcome {
    if (argv.len < 1) return .{ .usage = "usage" };
    var buf: [512]u8 = undefined;
    if (std.mem.eql(u8, argv[0], "check-flags")) {
        if (argv.len < 2) return .{ .usage = "check-flags needs a role" };
        const role = std.meta.stringToEnum(Role, argv[1]) orelse return .{ .usage = "check-flags: expert or attn" };
        if (role == .expert) { // the expert's flags are afd_expert's (qualification_cec3c04c)
            const a = afd_expert.parse(argv[2..]) catch |e| return .{ .usage = @errorName(e) };
            const mode = if (a.check_weights != null) "check-weights" else if (a.selftest != null) "selftest" else "serve";
            return .{ .ok = std.fmt.allocPrint(gpa, "check-flags expert {s} half {d} model {s} (afd_expert.parse)", .{ mode, a.half, a.model }) catch "check-flags expert" };
        }
        const f = parse(gpa, role, argv[2..]) catch |e| return .{ .usage = @errorName(e) };
        return .{ .ok = std.fmt.allocPrint(gpa, "check-flags {s}", .{describe(f, &buf)}) catch "check-flags" };
    }
    if (std.mem.eql(u8, argv[0], "check-audit")) return attn.checkAuditCmd(gpa, argv[1..]);
    const role = std.meta.stringToEnum(Role, argv[0]) orelse return .{ .usage = "usage" };
    const f = parse(gpa, role, argv[1..]) catch |e| return .{ .usage = @errorName(e) };
    std.debug.print("INFO {s}\n", .{describe(f, &buf)});
    if (f.role == .attn) return attn.run(gpa, io, f);
    return switch (f.verb) { // the expert's are afd_expert.command's (main): unreached from main
        .serve, .selftest, .@"check-weights" => .{ .usage = "expert: afd_expert.command" },
        .first, .teacher, .bench => .{ .usage = "BadVerb" },
    };
}

pub fn main(init: std.process.Init) !u8 {
    const all = try init.minimal.args.toSlice(init.arena.allocator());
    const argv = all[@min(all.len, 1)..];
    environ = init.environ_map; // attn serve's server reads it (qualification_4be15bbf)
    // the expert: qualification_cec3c04c's command end to end (its flags, its one line on how it ended, its exit code)
    if (argv.len > 0 and std.mem.eql(u8, argv[0], "expert")) return afd_expert.command(std.heap.smp_allocator, init.io, argv[1..]);
    const o = run(init.arena.allocator(), init.io, argv);
    if (o == .usage and std.mem.indexOf(u8, o.usage, "not-built") == null) std.debug.print("{s}", .{usage});
    var buf: [256]u8 = undefined;
    std.debug.print("{s}\n", .{o.line(&buf)});
    return o.code();
}

test "flags: each verb's required flags, ranges and defaults" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const e = try parse(a, .expert, &.{ "--half", "1", "--master", "192.0.2.10:18941", "--model", "/m", "--dev", "rocep1s0f0", "--layers", "8" });
    try std.testing.expectEqual(Verb.serve, e.verb);
    try std.testing.expectEqual(@as(u32, 1), e.half.?);
    try std.testing.expectEqual(@as(u16, 18941), e.master.?.getPort());
    try std.testing.expectEqual(@as(u32, 8), e.layers.?);
    const s = try parse(a, .expert, &.{ "selftest", "--selftest", "7", "--half", "0", "--model", "/m", "--out", "r.txt", "--rows", "1,4,8,2048", "--moe-layers", "3-7" });
    try std.testing.expectEqual(Verb.selftest, s.verb);
    try std.testing.expectEqual([2]u32{ 3, 7 }, s.moe_layers.?);
    try std.testing.expectEqualSlices(usize, &.{ 1, 4, 8, 2048 }, s.rows);
    const cw = try parse(a, .expert, &.{ "check-weights", "--half", "1", "--model", "/m", "--layers", "8", "--ref", "w.json" });
    try std.testing.expect(cw.verb == .@"check-weights" and cw.master == null and std.mem.eql(u8, cw.ref.?, "w.json"));
    const t = try parse(a, .attn, &.{ "teacher", "--master", "[::1]:18940", "--model", "/m", "--prompt", "p.json:A", "--teacher", "s.json", "--no-watcher", "--audit" });
    try std.testing.expect(t.verb == .teacher and !t.watcher and t.audit);
    const w = try parse(a, .attn, &.{ "check-weights", "--model", "/m", "--dry" });
    try std.testing.expect(w.verb == .@"check-weights" and w.dry);
    const b = try parse(a, .attn, &.{ "bench", "--master", "[::1]:18940", "--model", "/m", "--prompt", "p.json", "--bench", "decode=p1024:512,prefill=p1900,profile=32" });
    try std.testing.expect(b.verb == .bench and std.mem.eql(u8, b.bench.?, "decode=p1024:512,prefill=p1900,profile=32"));
    try std.testing.expectError(error.BenchIsBenchs, parse(a, .attn, &.{ "first", "--model", "/m", "--prompt", "p", "--dry", "--bench", "runs=1" }));
    const ex = std.testing.expectError;
    try ex(error.BadHalf, parse(a, .expert, &.{ "--half", "2", "--master", "1.2.3.4:1", "--model", "/m" }));
    try ex(error.NoMaster, parse(a, .expert, &.{ "--half", "0", "--model", "/m" }));
    try ex(error.BadMaster, parse(a, .expert, &.{ "--half", "0", "--master", "attention-host:18941", "--model", "/m" }));
    try ex(error.BadRows, parse(a, .expert, &.{ "--selftest", "1", "--half", "0", "--model", "/m", "--out", "x", "--rows", "4096" }));
    try ex(error.NoOut, parse(a, .expert, &.{ "--selftest", "1", "--half", "0", "--model", "/m" }));
    try ex(error.NoSeed, parse(a, .expert, &.{ "selftest", "--half", "0", "--model", "/m", "--out", "x" }));
    try ex(error.NoTeacher, parse(a, .attn, &.{ "teacher", "--master", "1.2.3.4:1", "--model", "/m", "--prompt", "p" }));
    try ex(error.MissingVerb, parse(a, .attn, &.{ "--model", "/m" }));
    try ex(error.BadVerb, parse(a, .attn, &.{ "selftest", "--model", "/m" }));
    try ex(error.BadVerb, parse(a, .expert, &.{ "first", "--model", "/m" }));
    try ex(error.UnknownFlag, parse(a, .expert, &.{ "--nope", "1" }));
    try ex(error.BadSpan, parseSpan("7-3"));
    const fe = try parse(a, .expert, &.{ "--half", "0", "--master", "1.2.3.4:5", "--model", "/m", "--fault", "exit:4" });
    try std.testing.expect(fe.fault.?.what == .exit and fe.fault.?.at == 4);
    try ex(error.BadFault, parseFault("kill:1"));
}

test "the summary line and the exit codes" {
    var buf: [128]u8 = undefined;
    const lost = fromFailure(.expert, .{ .why = .dead, .peer = 0, .seq = 4, .gave_up = 0, .breach = 0 });
    try std.testing.expectEqualStrings("AFD FAILED dead peer 0 seq 4", lost.line(&buf));
    try std.testing.expectEqual(@as(u8, 3), lost.code());
    const failed = fromFailure(.attn, .{ .why = .silent, .peer = 1, .seq = 5, .gave_up = 0, .breach = 0 });
    try std.testing.expectEqualStrings("AFD FAILED silent peer 1 seq 5", failed.line(&buf));
    try std.testing.expectEqual(@as(u8, 2), failed.code());
    try std.testing.expectEqualStrings("AFD OK first", (Outcome{ .ok = "first" }).line(&buf));
    try std.testing.expectEqual(@as(u8, 1), (Outcome{ .usage = "BadHalf" }).code());
    try std.testing.expectEqualStrings("AFD FAILED BadHalf peer - seq 0", (Outcome{ .usage = "BadHalf" }).line(&buf));
}

test "check-flags, check-audit's usage and the attention role's routes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = run(a, std.testing.io, &.{ "check-flags", "expert", "--half", "0", "--master", "192.0.2.10:18941", "--model", "/m", "--triton", "/t" });
    try std.testing.expectEqual(@as(u8, 0), ok.code());
    try std.testing.expectEqualStrings("check-flags expert serve half 0 model /m (afd_expert.parse)", ok.ok);
    try std.testing.expectEqual(@as(u8, 1), run(a, std.testing.io, &.{ "check-flags", "expert", "--half", "0", "--master", "192.0.2.10:18941", "--model", "/m" }).code()); // no --triton
    try std.testing.expectEqual(@as(u8, 1), run(a, std.testing.io, &.{ "check-flags", "expert", "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--check-weights", "f" }).code()); // two modes
    try std.testing.expectEqual(@as(u8, 1), run(a, std.testing.io, &.{ "check-flags", "attn", "first", "--model", "/m" }).code());
    try std.testing.expectEqual(@as(u8, 1), run(a, std.testing.io, &.{ "check-flags", "attn", "first", "--model", "/m", "--prompt", "p" }).code()); // no --master
    try std.testing.expectEqual(@as(u8, 0), run(a, std.testing.io, &.{ "check-flags", "attn", "first", "--model", "/m", "--prompt", "p", "--dry" }).code()); // --dry: no network
    try std.testing.expectEqualStrings("check-audit needs ATTN E0 E1", run(a, std.testing.io, &.{ "check-audit", "x.jsonl" }).usage);
    try std.testing.expectEqualStrings("NoFile", run(a, std.testing.io, &.{ "attn", "first", "--model", "/nonexistent", "--prompt", "/nonexistent/p.json", "--dry" }).usage);
    try std.testing.expectEqual(@as(u8, 1), run(a, std.testing.io, &.{"bogus"}).code());
}

test "the tool's entry compiles on the host; the attention role's tests (afd_attn.zig)" {
    _ = &main;
    _ = @import("afd_attn.zig");
    _ = @import("afd_attn_bench.zig");
    _ = @import("afd_attn_drafts.zig"); // qualification_21e21bc0: TF_AFD_MTP's model and attn bench's drafts
    _ = @import("afd_attn_graphs.zig"); // qualification_f16009a8: TF_AFD_GRAPHS on attn --dry (G-trace through the role)
    _ = @import("afd_cover_qmm.zig"); // qualification_18588626: G-cover CUDA, every qmm_group call built on the RTX 5090 and a GB10
    _ = @import("afd_expert_dry.zig");
    _ = @import("afd_attn_flight.zig"); // qualification_c533375f: attn --dry with TF_AFD_INFLIGHT=2, and its refusals
    _ = @import("afd_stripes.zig"); // qualification_17c70e37: TF_AFD_STRIPES in the roles
}
