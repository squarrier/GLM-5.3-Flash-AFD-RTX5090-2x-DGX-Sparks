//! sampling_tests: qualification_45c10bcf's attention-host nucleus draw; experts and wire settings are unchanged.
const std = @import("std");

pub fn parse(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadGlmSampling;
}

pub fn fromEnv() !bool {
    const v = std.c.getenv("TF_GLM_SAMPLING") orelse return false;
    return parse(std.mem.span(v));
}

test "sampling_tests sampling is default off and explicit one only" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse(""));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    for ([_][]const u8{ "2", "on", "true", " 1" }) |v| try std.testing.expectError(error.BadGlmSampling, parse(v));
}
