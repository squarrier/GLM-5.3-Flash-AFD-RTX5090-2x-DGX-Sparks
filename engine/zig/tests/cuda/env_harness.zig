//! The tests' environment harness (FJPANIC): a test that changes the process's environment through libc keeps std's
//! startup environment whole. std captured the startup envp array at start as a fixed-length slice
//! (std.process.Environ.PosixBlock) and reads it lazily (Io.Threaded.scanEnviron unwraps every entry). libc's unsetenv
//! shifts the later entries of `environ` down in place, and until a setenv adds a name `environ` is that startup array:
//! unsetting a name the process started with left a null inside std's slice. Under the build runner nothing scans
//! before a test, so the test's first stderr write panicked "attempt to use null value" holding the debug Io's environ
//! mutex, and the panic handler's debug-info lookup waited on that mutex forever (joined_profile mp-full-on, TF_GLM_SAMPLING=1).
//! This file: Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The fresh names `detach` may add and remove (the first one the process does not have).
const fresh = [_][*:0]const u8{ "TF_TEST_ENV_DETACH", "TF_TEST_ENV_DETACH_1", "TF_TEST_ENV_DETACH_2", "TF_TEST_ENV_DETACH_3" };

fn startup() [:null]const ?[*:0]const u8 {
    return std.testing.environ.block.slice;
}

/// libc's `environ` is an array of libc's own, not std's startup array.
pub fn detached() bool {
    return @intFromPtr(std.c.environ) != @intFromPtr(startup().ptr);
}

/// Before a test's first setenv/unsetenv: adding a name the process lacks makes libc copy `environ` into its own array
/// (a new entry never fits the startup array); removing it again leaves the environment as it was.
pub fn detach() error{EnvironNotDetached}!void {
    if (detached()) return;
    for (fresh) |name| {
        if (std.c.getenv(name) != null) continue;
        if (setenv(name, "1", 1) != 0) break;
        _ = unsetenv(name);
        break;
    }
    if (!detached()) return error.EnvironNotDetached;
}

/// std's startup slice holds no null entry (each one is unwrapped when std scans the environment).
pub fn startupWhole() bool {
    for (startup()) |entry| if (entry == null) return false;
    return true;
}

/// The name of the process's first startup entry, copied into `buf` (null: no startup environment).
pub fn startupName(buf: []u8) ?[:0]const u8 {
    const s = startup();
    if (s.len == 0) return null;
    const entry = std.mem.span(s[0] orelse return null);
    const n = std.mem.indexOfScalar(u8, entry, '=') orelse entry.len;
    if (n == 0 or n >= buf.len) return null;
    @memcpy(buf[0..n], entry[0..n]);
    buf[n] = 0;
    return buf[0..n :0];
}
