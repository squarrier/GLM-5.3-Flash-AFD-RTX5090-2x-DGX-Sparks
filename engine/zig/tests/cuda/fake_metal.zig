//! Host memory in place of the Metal runtime's buffers (mapped files as device.zig maps them): weights.zig loads on any CPU.
const std = @import("std");

pub const ResourceOptions = struct {
    pub const shared: usize = 0 << 4;
    pub const untracked: usize = 1 << 8;
};

pub const Buffer = struct {
    id: [*]u8,
    len: usize = 0,

    pub fn contents(self: Buffer) [*]u8 {
        return self.id;
    }

    pub fn deinit(self: Buffer) void {
        std.heap.page_allocator.free(self.id[0..self.len]);
    }
};

pub const Device = struct {
    /// Zeroed pages of `len` bytes, as a fresh shared Metal buffer reads.
    pub fn buffer(_: Device, len: usize, _: usize) !Buffer {
        const m = try std.heap.page_allocator.alloc(u8, len);
        @memset(m, 0);
        return .{ .id = m.ptr, .len = len };
    }
};

/// A read-only file mapped at a page boundary, its length rounded up to whole pages (device.zig's).
pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) u8,
    size: usize,

    pub fn open(path: [*:0]const u8) !MappedFile {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end <= 0) return error.EmptyFile;
        const size: usize = @intCast(end);
        const page = std.heap.pageSize();
        const len = (size + page - 1) / page * page;
        const bytes = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        return .{ .bytes = bytes, .size = size };
    }

    pub fn deinit(self: MappedFile) void {
        std.posix.munmap(self.bytes);
    }
};
