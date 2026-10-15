//! literal_boundary (local integration): Mia GLM #158's literal-closer cases on the server's text paths, one function at a time (CPU, no sockets).
//! Reimplemented from MiaAI-Lab GLM PR #158 (@rwl4, 029a852f) tests/tool_envelope_contract.py's cases; no code copied.
//! Every check prints one `literal_boundary {json}` line and the test passes; `-Dstrict=true` fails it on any `boundary` check that fails.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const clock = @import("clock.zig");
const options = @import("literal_boundary_options");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const write_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":{\"type\":\"object\",\"properties\":{\"content\":{\"type\":\"string\"}},\"required\":[\"content\"]}}}]";
pub const file_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}]";

/// Mia #158's five values, in her order.
pub const values = [_][]const u8{
    "hello",
    "before </tool_call> AFTER_IN_FILE",
    "a </tool_call> b </tool_call> c",
    "<tool_call>this is literal file content</tool_call>",
    "unicode \u{3bb} \u{1f600}\nquotes \" and backslash \\ </tool_call> tail",
};

pub fn call(a: Allocator, v: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "<tool_call>write<arg_key>content</arg_key><arg_value>", v, "</arg_value></tool_call>" });
}

/// Bytes of `s` cut back to a whole UTF-8 character.
pub fn clip(s: []const u8, n: usize) []const u8 {
    if (s.len <= n) return s;
    var cut = n;
    while (cut > 0 and s[cut] & 0xc0 == 0x80) cut -= 1;
    return s[0..cut];
}

pub const Rec = struct {
    a: Allocator,
    suite: []const u8,
    total: usize = 0,
    failed: usize = 0,
    boundary_failed: usize = 0,

    pub fn check(r: *Rec, case: []const u8, path: []const u8, site: []const u8, kind: []const u8, pass: bool, got: []const u8) void {
        r.total += 1;
        if (!pass) {
            r.failed += 1;
            if (std.mem.eql(u8, kind, "boundary")) r.boundary_failed += 1;
        }
        const q: []const u8 = json.quote(r.a, clip(got, 300), .{ .ascii = true }) catch "\"(quote failed)\"";
        std.debug.print("literal_boundary {{\"suite\":\"{s}\",\"case\":\"{s}\",\"path\":\"{s}\",\"site\":\"{s}\",\"kind\":\"{s}\",\"pass\":{},\"got\":{s}}}\n", .{ r.suite, case, path, site, kind, pass, q });
    }

    pub fn done(r: *Rec) !void {
        std.debug.print("literal_boundary-TOTAL {{\"suite\":\"{s}\",\"checks\":{d},\"failed\":{d},\"boundary_failed\":{d},\"strict\":{}}}\n", .{ r.suite, r.total, r.failed, r.boundary_failed, options.strict });
        if (options.strict and r.boundary_failed > 0) return error.literal_boundary_boundaryfailed;
    }
};

pub fn tools(a: Allocator, text: []const u8) ![]const Value {
    return (try json.parse(a, text)).ok.array;
}

/// The arguments' key as a string, or null when the call has no such key.
pub fn argOf(a: Allocator, c: Value, key: []const u8) !?[]const u8 {
    const args = c.get("function").?.strField("arguments") orelse return null;
    const parsed = switch (try json.parse(a, args)) {
        .ok => |v| v,
        .err => return null,
    };
    return parsed.strField(key);
}

fn parsedSummary(a: Allocator, p: tool_parse.Parsed) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const n = if (p.calls) |cs| cs.len else 0;
    try out.print(a, "calls={d}", .{n});
    if (p.calls) |cs| for (cs) |c| {
        try out.print(a, " {s}({s})", .{ c.get("function").?.strField("name").?, c.get("function").?.strField("arguments").? });
    };
    try out.print(a, " content={s}", .{p.content});
    return out.items;
}

fn oneCall(a: Allocator, p: tool_parse.Parsed, key: []const u8, want: []const u8) !bool {
    const cs = p.calls orelse return false;
    if (cs.len != 1) return false;
    const got = try argOf(a, cs[0], key) orelse return false;
    return std.mem.eql(u8, got, want);
}

test "literal_boundary units: final parse, content hiding, think split, one-call filter, streamer, closeCall, large, extras" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: Rec = .{ .a = a, .suite = "units" };
    const w = try tools(a, write_tools);
    const markers = reply_text.think_markers;
    for (values, 0..) |v, vi| {
        const raw = try call(a, v);
        const text = try std.mem.concat(a, u8, &.{ "Before. ", raw, " After." });
        const case = try std.fmt.allocPrint(a, "v{d}", .{vi});
        // final (non-streamed) parse, both call policies
        for ([_]?usize{ null, 1 }) |mc| {
            const p = try tool_parse.parse(a, text, w, mc);
            const pass = (try oneCall(a, p, "content", v)) and std.mem.eql(u8, p.content, "Before.  After.");
            r.check(case, if (mc == null) "final-parse" else "final-parse-single", "tool_parse.zig:55-58 envelopes; :371-407 glmCall", "boundary", pass, try parsedSummary(a, p));
        }
        // content hiding while streaming: every byte prefix of the call shows nothing
        var leaks: usize = 0;
        var first: ?usize = null;
        var shown: []const u8 = "";
        for (0..raw.len + 1) |i| {
            const h = try reply_text.hideToolCalls(a, raw[0..i], false);
            if (h.len != 0) {
                leaks += 1;
                if (first == null) {
                    first = i;
                    shown = h;
                }
            }
        }
        r.check(case, "hide-stream-prefixes", "reply_text.zig:127-128 hideInto (first closer)", "boundary", leaks == 0, try std.fmt.allocPrint(a, "{d}/{d} prefixes leak; first at byte {?d}: {s}", .{ leaks, raw.len + 1, first, shown }));
        // content hiding at the end of the reply
        const fin = try reply_text.hideToolCalls(a, text, true);
        r.check(case, "hide-finished", "reply_text.zig:127-128 hideInto (first closer)", "boundary", std.mem.eql(u8, fin, "Before.  After."), fin);
        // think split while the call streams inside an unclosed think block (Mia: ThinkSplit holds it)
        var think_bad: usize = 0;
        var think_first: []const u8 = "";
        for (0..raw.len) |i| {
            const s = try reply_text.splitThinking(a, try std.mem.concat(a, u8, &.{ "thought ", raw[0..i] }), false, markers);
            if (s.answer.len != 0 or !std.mem.startsWith(u8, "thought ", s.reasoning)) {
                if (think_bad == 0) think_first = try std.fmt.allocPrint(a, "at {d}: reasoning={s} answer={s}", .{ i, s.reasoning, s.answer });
                think_bad += 1;
            }
        }
        r.check(case, "think-split-prefixes", "reply_text.zig:53-94 splitThinking", "boundary", think_bad == 0, try std.fmt.allocPrint(a, "{d} bad {s}", .{ think_bad, think_first }));
        // a call ending an unclosed think block (finished): the call is the answer and parses whole
        {
            const s = try reply_text.splitThinking(a, try std.mem.concat(a, u8, &.{ "thought ", raw }), true, markers);
            const split_ok = std.mem.eql(u8, s.reasoning, "thought ") and std.mem.eql(u8, s.answer, raw);
            r.check(case, "think-unclosed-split", "reply_text.zig:77-87 splitThinking", "boundary", split_ok, try std.fmt.allocPrint(a, "reasoning={s} answer={s}", .{ s.reasoning, s.answer }));
            const p = try tool_parse.parse(a, s.answer, w, null);
            r.check(case, "think-unclosed-parse", "tool_parse.zig:55-58 envelopes", "boundary", (try oneCall(a, p, "content", v)) and p.content.len == 0, try parsedSummary(a, p));
        }
        // Mia's ThinkSplit contract: a call followed only by </think> is the answer (ours keeps it as reasoning)
        {
            const s = try reply_text.splitThinking(a, try std.mem.concat(a, u8, &.{ "thought ", raw, "</think>" }), true, markers);
            r.check(case, "think-call-then-close", "reply_text.zig:76 splitThinking (no think-call promotion)", "contract", std.mem.eql(u8, s.reasoning, "thought ") and std.mem.eql(u8, s.answer, raw), try std.fmt.allocPrint(a, "reasoning={s} answer={s}", .{ s.reasoning, s.answer }));
        }
        // the one-call policy's text filter (parallel_tool_calls false), whole and one byte at a time
        {
            const single: tool_stream.Policy = .{ .single = true };
            const whole = try single.content(a, text);
            r.check(case, "single-filter-whole", "tool_stream.zig:216-227 TextFilter alone (served text reaches it after hideInto)", "isolated", std.mem.eql(u8, whole, "Before.  After."), whole);
            var f: tool_stream.TextFilter = .{};
            var got: std.ArrayList(u8) = .empty;
            var i: usize = 0;
            while (i < text.len) : (i += 1) try got.appendSlice(a, try f.feed(a, text[i .. i + 1]));
            try got.appendSlice(a, f.finish());
            r.check(case, "single-filter-bytes", "tool_stream.zig:216-227 TextFilter alone (served text reaches it after hideInto)", "isolated", std.mem.eql(u8, got.items, "Before.  After."), got.items);
        }
        // the streamed parser: GLM's NAME<arg_key> body is not its format, so it sends nothing
        {
            var s = try tool_stream.Streamer.init(a, w);
            var out: std.ArrayList(Value) = .empty;
            for (0..text.len + 1) |i| try s.feed(text[0..i], &out);
            r.check(case, "streamer-glm", "tool_stream.zig:95-98 Streamer .call (abort)", "na", out.items.len == 0 and s.state == .abort, try std.fmt.allocPrint(a, "deltas={d} state={s}", .{ out.items.len, @tagName(s.state) }));
        }
    }
    // closeCall: the end token came before </tool_call>, and the value holds a literal closer
    {
        const open = "x <tool_call>write<arg_key>content</arg_key><arg_value>a </tool_call> b</arg_value>";
        const close = try tool_parse.closeCall(a, open, w);
        r.check("close", "close-call", "tool_parse.zig:165-167 closeCall (first closer pairs)", "boundary", std.mem.eql(u8, close, "</tool_call>"), close);
        const p = try tool_parse.parse(a, try std.mem.concat(a, u8, &.{ open, close }), w, null);
        r.check("close", "close-call-parse", "tool_parse.zig:55-58 envelopes", "boundary", (try oneCall(a, p, "content", "a </tool_call> b")) and std.mem.eql(u8, p.content, "x"), try parsedSummary(a, p));
    }
    // two calls, the first holding a literal closer
    {
        const raw = try std.mem.concat(a, u8, &.{ try call(a, "a </tool_call> b"), "\n", try call(a, "second") });
        const p = try tool_parse.parse(a, raw, w, null);
        var ok = p.calls != null and p.calls.?.len == 2 and p.content.len == 0;
        if (ok) ok = std.mem.eql(u8, (try argOf(a, p.calls.?[0], "content")) orelse "", "a </tool_call> b") and std.mem.eql(u8, (try argOf(a, p.calls.?[1], "content")) orelse "", "second");
        r.check("two", "two-calls-parse", "tool_parse.zig:55-58 envelopes", "boundary", ok, try parsedSummary(a, p));
        const h = try reply_text.hideToolCalls(a, raw, true);
        r.check("two", "two-calls-hide", "reply_text.zig:127-128 hideInto", "boundary", reply_text.pyStrip(h).len == 0, h);
    }
    // truncated values: no call may come out of a cut reply
    {
        const cut1 = try call(a, "a </tool_call> b");
        const cuts = [_][]const u8{ cut1[0 .. cut1.len - "</arg_value></tool_call>".len], "<tool_call>write<arg_key>content</arg_key><arg_value>cut" };
        for (cuts, 0..) |cut, ci| {
            const case = try std.fmt.allocPrint(a, "cut{d}", .{ci});
            const p = try tool_parse.parse(a, cut, w, null);
            r.check(case, "cut-parse", "tool_parse.zig:55-58 envelopes", "boundary", p.calls == null, try parsedSummary(a, p));
            const h = try reply_text.hideToolCalls(a, cut, true);
            r.check(case, "cut-hide", "reply_text.zig:127-128 hideInto", "boundary", h.len == 0, h);
        }
    }
    // Mia's fallback formats still parse (JSON payload, Qwen XML)
    for ([_][]const u8{ "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"JSON\"}}</tool_call>", "<tool_call><function=write><parameter=content>Qwen</parameter></function></tool_call>" }, 0..) |raw, i| {
        const p = try tool_parse.parse(a, raw, w, null);
        r.check(try std.fmt.allocPrint(a, "fmt{d}", .{i}), "fallback-format", "tool_parse.zig:245-292 payload", "boundary", p.content.len == 0 and p.calls != null and p.calls.?.len == 1, try parsedSummary(a, p));
    }
    // Qwen XML with a literal closer in a parameter: the streamer is value-aware, the final parse is not
    {
        const raw = "<tool_call>\n<function=write>\n<parameter=content>\nbefore </tool_call> AFTER_IN_FILE\n</parameter>\n</function>\n</tool_call>";
        var s = try tool_stream.Streamer.init(a, w);
        var out: std.ArrayList(Value) = .empty;
        for (0..raw.len + 1) |i| try s.feed(raw[0..i], &out);
        var sent: std.ArrayList(u8) = .empty;
        for (out.items) |d| for (d.get("tool_calls").?.array) |one| {
            if (one.get("function")) |f| if (f.strField("arguments")) |t| try sent.appendSlice(a, t);
        };
        const sent_val = switch (try json.parse(a, sent.items)) {
            .ok => |v| v.strField("content") orelse "",
            .err => "",
        };
        r.check("qwen", "qwen-streamer", "tool_stream.zig:144-172 Streamer .value (value-aware)", "info", std.mem.eql(u8, sent_val, "before </tool_call> AFTER_IN_FILE"), sent.items);
        const p = try tool_parse.parse(a, raw, w, null);
        r.check("qwen", "qwen-final-parse", "tool_parse.zig:55-58 envelopes", "info", try oneCall(a, p, "content", "before </tool_call> AFTER_IN_FILE"), try parsedSummary(a, p));
    }
    // large cumulative answer: hiding and parsing stay correct and bounded (Mia: < 3 s)
    {
        var body: std.ArrayList(u8) = .empty;
        for (0..12000) |_| try body.appendSlice(a, "source line </tool_call> tail\n");
        const big = try call(a, body.items);
        const t0 = clock.nowNs(std.testing.io);
        const h = try reply_text.hideToolCalls(a, big, true);
        const p = try tool_parse.parse(a, big, w, null);
        const secs = clock.seconds(clock.nowNs(std.testing.io) - t0);
        r.check("large", "large-hide", "reply_text.zig:127-128 hideInto", "boundary", h.len == 0, try std.fmt.allocPrint(a, "bytes={d} shown={d}", .{ big.len, h.len }));
        r.check("large", "large-parse", "tool_parse.zig:55-58 envelopes", "boundary", try oneCall(a, p, "content", body.items), try std.fmt.allocPrint(a, "calls={d} first_args_len={d}", .{ if (p.calls) |cs| cs.len else 0, if (p.calls) |cs| cs[0].get("function").?.strField("arguments").?.len else 0 }));
        r.check("large", "large-seconds", "hide+parse wall time", "boundary", secs < 3.0, try std.fmt.allocPrint(a, "{d:.3} s", .{secs}));
    }
    // extras beyond Mia's set
    {
        // a two-argument write_file: the value with the closer is silently dropped, the other kept
        const fw = try tools(a, file_tools);
        const raw = "<tool_call>write_file<arg_key>path</arg_key><arg_value>zig/src/server/x.zig</arg_value><arg_key>content</arg_key><arg_value>const close = \"</tool_call>\";\n</arg_value></tool_call>";
        const p = try tool_parse.parse(a, raw, fw, null);
        const path_ok = if (p.calls) |cs| (cs.len == 1 and std.mem.eql(u8, (try argOf(a, cs[0], "path")) orelse "", "zig/src/server/x.zig")) else false;
        const content_ok = if (p.calls) |cs| (cs.len == 1 and std.mem.eql(u8, (try argOf(a, cs[0], "content")) orelse "", "const close = \"</tool_call>\";\n")) else false;
        r.check("x-writefile", "two-arg-parse", "tool_parse.zig:55-58 envelopes; :390 glmCall drops a value without </arg_value>", "boundary", path_ok and content_ok and reply_text.pyStrip(p.content).len == 0, try parsedSummary(a, p));
        // a literal </think> inside a value, call in an unclosed think block
        const tv = "x </think> y";
        const s = try reply_text.splitThinking(a, try std.mem.concat(a, u8, &.{ "thought ", try call(a, tv) }), true, markers);
        r.check("x-think", "think-close-in-value-unclosed", "reply_text.zig:76 splitThinking finds </think> inside the value", "extra", std.mem.eql(u8, s.reasoning, "thought ") and std.mem.eql(u8, s.answer, try call(a, tv)), try std.fmt.allocPrint(a, "reasoning={s} answer={s}", .{ s.reasoning, s.answer }));
        const s2 = try reply_text.splitThinking(a, try std.mem.concat(a, u8, &.{ "brief</think>Before. ", try call(a, tv) }), true, markers);
        r.check("x-think", "think-close-in-value-closed", "reply_text.zig:76 (first </think> is the real one)", "extra", std.mem.eql(u8, s2.reasoning, "brief"), try std.fmt.allocPrint(a, "reasoning={s} answer={s}", .{ s2.reasoning, s2.answer }));
        // the reverse: structured JSON content holding literal tool markup in a string, tools offered
        const js = "{\"note\":\"<tool_call>x</tool_call>\"}";
        const pj = try tool_parse.parse(a, js, w, null);
        r.check("x-json", "json-string-markup-parallel", "tool_parse.zig:148-151 (left as text)", "extra", pj.calls == null and std.mem.eql(u8, pj.content, js), try parsedSummary(a, pj));
        const pj1 = try tool_parse.parse(a, js, w, 1);
        r.check("x-json", "json-string-markup-single", "tool_parse.zig:150 (dropped from a one-call reply)", "extra", pj1.calls == null and std.mem.eql(u8, pj1.content, js), try parsedSummary(a, pj1));
        const hj = try reply_text.hideToolCalls(a, js, true);
        r.check("x-json", "json-string-markup-stream-hide", "reply_text.zig:97-130 hideInto", "extra", std.mem.eql(u8, hj, js), hj);
        // a GLM value that never writes </arg_value> before the closer: today a call with that value dropped; under
        // Mia's rule (an open value holds the call) no call at all. Recorded: the one deliberate output change outside
        // well-formed literal-closer values.
        const mal = "<tool_call>write<arg_key>content</arg_key><arg_value>v</tool_call>";
        const pm = try tool_parse.parse(a, mal, w, null);
        r.check("x-malformed", "value-without-close", "tool_parse.zig:55-58 envelopes + :390 glmCall", "change", pm.calls == null, try parsedSummary(a, pm));
        // Hermes/Qwen JSON payload with a literal closer inside a JSON string: the same bug class in another format
        const hj2 = "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"a </tool_call> b\"}}</tool_call>";
        const ph = try tool_parse.parse(a, hj2, w, null);
        r.check("x-jsonfmt", "json-payload-closer-in-string", "tool_parse.zig:55-58 envelopes (JSON strings are not GLM values)", "extra", try oneCall(a, ph, "content", "a </tool_call> b"), try parsedSummary(a, ph));
    }
    try r.done();
}
