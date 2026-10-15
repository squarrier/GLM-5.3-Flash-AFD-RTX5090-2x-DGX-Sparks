//! Ordinary-build reachability probe; compile only, never call transport.
const std = @import("std");
const server = @import("server.zig");
const http = @import("http_conn.zig");
const routes = @import("routes.zig");
comptime {
    if (@import("builtin").is_test) @compileError("ordinary object required");
    if (@import("sse_test_observe.zig").Test != void) @compileError("observer type survived");
    for (.{ @import("openai.zig"), @import("chat.zig"), @import("anthropic.zig"), @import("responses.zig") }) |owner| {
        if (@hasDecl(owner, "TestFixture")) if (owner.TestFixture != void) @compileError("owner fixture survived ordinary build");
    }
}
export fn qualification_5753d72e_dispatch(srv: *server.Server, conn: *http.Conn) void {
    routes.dispatch(srv, conn, conn.gpa);
}
