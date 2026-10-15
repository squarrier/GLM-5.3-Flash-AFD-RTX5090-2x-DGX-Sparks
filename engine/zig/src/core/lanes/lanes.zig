//! The lane engine's host core (Python lane_engine.py); backends run forwards and draws behind `backend.Backend`.
const std = @import("std");

pub const table = @import("table.zig");
pub const allocate = @import("allocate.zig");
pub const accept = @import("accept.zig");
pub const proposer = @import("proposer.zig");
pub const sampling = @import("sampling.zig");
pub const fill = @import("fill.zig");
pub const shape = @import("shape.zig");
pub const plan_lanes = @import("plan_lanes.zig");
pub const gpu_rule = @import("gpu_rule.zig");
pub const gpu_full = @import("gpu_full.zig");
pub const depth = @import("depth.zig");
pub const config = @import("config.zig");
pub const stream = @import("stream.zig");
pub const grammar_window = @import("grammar_window.zig");
pub const grammar = @import("grammar.zig");
/// grammar_mask: request-owned Rules over a real library matcher (grammar_abi's tfg_owned shape)
pub const grammar_owned = @import("grammar_owned.zig");
pub const backend = @import("backend.zig");
pub const events = @import("events.zig");
pub const windows = @import("windows.zig");
pub const trail = @import("trail.zig");
pub const engine = @import("engine.zig");
pub const fake = @import("fake.zig");

pub const Engine = engine.Engine;
pub const Config = config.Config;
pub const Model = config.Model;
pub const Stream = stream.Stream;
pub const Sampling = sampling.Sampling;
pub const SuffixLookup = proposer.SuffixLookup;

test {
    std.testing.refAllDecls(@This());
    _ = @import("engine_test.zig");
    _ = @import("open_test.zig"); // prefill_interleave: a prompt pass a step at a time (#623's fillStream)
    _ = @import("gpu_full_test.zig");
}

pub const structured_control = @import("structured_control.zig");
pub const structured = @import("structured.zig");
