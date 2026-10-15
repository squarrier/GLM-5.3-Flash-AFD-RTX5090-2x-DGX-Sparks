//! Compile-time fixture observations; no namespace, hook or TLS in ordinary builds.
const std = @import("std");
pub const Test = if (@import("builtin").is_test) struct {
    pub const SessionPhase = enum { start, wrote, comment, stop };
    pub const Edge = enum { begin, end };
    pub const Route = enum { messages, responses };
    pub const Phase = enum { enter, exit };
    pub const Event = union(enum) {
        run_enter: void,
        run_open: void,
        run_exit: void,
        sink: struct { beat_present: bool, idle_present: bool },
        generated: void,
        session: struct { phase: SessionPhase, active: bool, next_ns: i96, attached: bool },
        generate_enter: struct { sink_present: bool, idle_present: bool },
        request_end: void,
        box_deinit: struct { phase: Edge, finished: bool },
        drain_begin: void,
        drain_end: struct { finished: bool },
        idle: struct { mutex: *std.Io.Mutex, io: std.Io, finished_snapshot: bool, idle_present: bool },
        translated: struct { route: Route, phase: Phase, sequence_or_index: i64, terminal: bool },
    };
    pub const Hook = struct { ctx: *anyopaque, record: *const fn (*anyopaque, Event) void };
    pub threadlocal var active: ?*Hook = null;
    pub fn send(hook: ?*Hook, event: Event) void {
        if (hook) |h| h.record(h.ctx, event);
    }
    pub fn record(event: Event) void { send(active, event); }
} else void;
