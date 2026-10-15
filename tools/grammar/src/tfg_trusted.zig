//! grammar_abi trusted loader for the release libtfgrammar ABI (tf_grammar.cc at TensorFold v1.0.6 c080a18e).
//! One operator-given absolute path; the opened inode's owner, mode, parent, sha256, ELF header and .dynsym (all 12
//! release symbols) are checked BEFORE dlopen, so a wrong library never runs its initializers; that same inode is
//! then dlopen'ed through /proc/self/fd with RTLD_NODELETE and is never unloaded (x86-run-2: dlclose of a refused
//! C++ libtfgrammar crashed the process at exit). The first outcome is sticky for the process. No library function
//! is called here. Claude Opus 5.5 via Hermes (local integration).
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// The release binding's table (core/lanes/grammar.zig:7-20 at c080a18e), field for field.
pub const Fns = struct {
    open: *const fn ([*]const u8, usize, c_int, [*]const i32, c_int, [*]u8, usize) callconv(.c) ?*anyopaque,
    close: *const fn (*anyopaque) callconv(.c) void,
    words: *const fn (*anyopaque) callconv(.c) c_int,
    compile: *const fn (*anyopaque, c_int, [*]const u8, usize, [*]u8, usize) callconv(.c) ?*anyopaque,
    free: *const fn (*anyopaque) callconv(.c) void,
    matcher: *const fn (*anyopaque) callconv(.c) ?*anyopaque,
    matcher_free: *const fn (*anyopaque) callconv(.c) void,
    accept: *const fn (*anyopaque, i32) callconv(.c) c_int,
    rollback: *const fn (*anyopaque, c_int) callconv(.c) c_int,
    terminated: *const fn (*anyopaque) callconv(.c) c_int,
    fill: *const fn (*anyopaque, [*]i32, c_int) callconv(.c) c_int,
    drafts: *const fn (*anyopaque, [*]i32, c_int) callconv(.c) c_int,
};

/// Why a load was refused; the first one is kept for the process.
pub const Failure = enum { bad_digest_arg, relative_path, open_failed, not_regular, unsafe_owner, unsafe_mode, unsafe_parent, read_failed, digest_mismatch, not_elf, wrong_arch, dlopen_failed, missing_symbol };

pub const Report = struct {
    failure: ?Failure = null,
    errno: u16 = 0,
    symbol: []const u8 = "",
    machine: u16 = 0,
    sha256: [64]u8 = @splat('0'),
    hashed: bool = false,
    elf_symbols: bool = false, // all 12 found in .dynsym before dlopen
    dlopened: bool = false,
    dlclosed: bool = false, // never set since x86-run-2: a C++ library is not unloaded once its initializers ran
    pinned: bool = false, // dlsym failed after the .dynsym check passed: kept loaded and unused
    detail_buf: [160]u8 = undefined,
    detail_len: usize = 0,

    pub fn detail(r: *const Report) []const u8 {
        return r.detail_buf[0..r.detail_len];
    }
};

/// The ELF machine this process runs as (EM_X86_64 62, EM_AARCH64 183).
pub const host_machine: u16 = switch (builtin.cpu.arch) {
    .x86_64 => 62,
    .aarch64 => 183,
    else => @compileError("grammar_abi loader: x86_64 or aarch64 only"),
};

pub const Loader = struct {
    mutex: std.atomic.Mutex = .unlocked,
    state: enum { unloaded, loaded, failed } = .unloaded,
    fns: Fns = undefined,
    handle: ?*anyopaque = null,
    report: Report = .{},
    attempts: u32 = 0, // load procedures run: at most one per Loader (sticky)
    refused_after: u32 = 0, // later requests answered from the cached failure

    /// The library's table, or null with `report` saying why; only the first call does any work.
    pub fn get(l: *Loader, path: []const u8, digest_hex: []const u8) ?*const Fns {
        while (!l.mutex.tryLock()) std.atomic.spinLoopHint();
        defer l.mutex.unlock();
        switch (l.state) {
            .loaded => return &l.fns,
            .failed => {
                l.refused_after += 1;
                return null;
            },
            .unloaded => {},
        }
        l.attempts += 1;
        l.state = .failed; // sticky from the start, as the release binding's lib_failed
        if (!l.attempt(path, digest_hex)) return null;
        l.state = .loaded;
        return &l.fns;
    }

    fn fail(l: *Loader, why: Failure, errno: u16, comptime fmt: []const u8, args: anytype) bool {
        l.report.failure = why;
        l.report.errno = errno;
        const d = std.fmt.bufPrint(&l.report.detail_buf, fmt, args) catch l.report.detail_buf[0..0];
        l.report.detail_len = d.len;
        return false;
    }

    fn attempt(l: *Loader, path: []const u8, digest_hex: []const u8) bool {
        var want: [64]u8 = undefined;
        if (digest_hex.len != 64) return l.fail(.bad_digest_arg, 0, "digest must be 64 hex characters", .{});
        for (digest_hex, 0..) |ch, i| want[i] = switch (ch) {
            '0'...'9', 'a'...'f' => ch,
            'A'...'F' => ch + 32,
            else => return l.fail(.bad_digest_arg, 0, "digest must be hex", .{}),
        };
        if (!absoluteClean(path)) return l.fail(.relative_path, 0, "not an absolute normalized path: {s}", .{path});
        var zbuf: [4096]u8 = undefined;
        @memcpy(zbuf[0..path.len], path);
        zbuf[path.len] = 0;
        const zpath: [*:0]const u8 = @ptrCast(&zbuf);

        const rc = linux.open(zpath, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, 0);
        if (errnoOf(rc)) |e| return l.fail(.open_failed, e, "open errno {d}", .{e});
        const fd: i32 = @intCast(rc);
        defer _ = linux.close(fd);

        var st: linux.Statx = undefined;
        const mask: linux.STATX = .{ .TYPE = true, .MODE = true, .UID = true, .SIZE = true };
        if (errnoOf(linux.statx(fd, "", linux.AT.EMPTY_PATH, mask, &st))) |e| return l.fail(.open_failed, e, "statx errno {d}", .{e});
        if (st.mode & linux.S.IFMT != linux.S.IFREG) return l.fail(.not_regular, 0, "not a regular file (mode {o})", .{st.mode});
        const euid = linux.geteuid();
        if (st.uid != euid and st.uid != 0) return l.fail(.unsafe_owner, 0, "owner uid {d} is neither {d} nor root", .{ st.uid, euid });
        if (st.mode & (linux.S.IWGRP | linux.S.IWOTH) != 0) return l.fail(.unsafe_mode, 0, "group/other writable (mode {o})", .{st.mode & 0o7777});
        if (!l.parentSafe(path, euid)) return false;

        var h = std.crypto.hash.sha2.Sha256.init(.{});
        var head: [64]u8 = @splat(0);
        var head_len: usize = 0;
        var buf: [1 << 16]u8 = undefined;
        while (true) {
            const n = linux.read(fd, &buf, buf.len);
            if (errnoOf(n)) |e| return l.fail(.read_failed, e, "read errno {d}", .{e});
            if (n == 0) break;
            if (head_len < head.len) {
                const k = @min(head.len - head_len, n);
                @memcpy(head[head_len..][0..k], buf[0..k]);
                head_len += k;
            }
            h.update(buf[0..n]);
        }
        var digest: [32]u8 = undefined;
        h.final(&digest);
        l.report.sha256 = hexLower(digest);
        l.report.hashed = true;
        if (!std.mem.eql(u8, &l.report.sha256, &want)) return l.fail(.digest_mismatch, 0, "sha256 differs from the manifest", .{});

        // ELF64, little-endian, ET_DYN, this machine
        if (head_len < 64 or !std.mem.eql(u8, head[0..4], "\x7fELF")) return l.fail(.not_elf, 0, "no ELF magic", .{});
        if (head[4] != 2 or head[5] != 1) return l.fail(.not_elf, 0, "not ELF64 little-endian (class {d} data {d})", .{ head[4], head[5] });
        const e_type = std.mem.readInt(u16, head[16..18], .little);
        l.report.machine = std.mem.readInt(u16, head[18..20], .little);
        if (e_type != 3) return l.fail(.not_elf, 0, "e_type {d} is not ET_DYN", .{e_type});
        if (l.report.machine != host_machine) return l.fail(.wrong_arch, 0, "e_machine {d}, this process is {d}", .{ l.report.machine, host_machine });
        // every required symbol, from the verified file's .dynsym, before any of the library's code can run
        if (!l.elfSymbols(fd)) return false;

        // exactly the inode hashed above; NODELETE: once its initializers have run, the library is never unmapped
        var pbuf: [64]u8 = undefined;
        const proc = std.fmt.bufPrintSentinel(&pbuf, "/proc/self/fd/{d}", .{fd}, 0) catch unreachable;
        const handle = std.c.dlopen(proc.ptr, .{ .NOW = true, .NODELETE = true }) orelse {
            const why = if (std.c.dlerror()) |m| std.mem.sliceTo(m, 0) else "unknown";
            return l.fail(.dlopen_failed, 0, "dlopen: {s}", .{why[0..@min(why.len, 120)]});
        };
        l.report.dlopened = true;
        var f: Fns = undefined;
        const info = @typeInfo(Fns).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            const sym = std.c.dlsym(handle, "tfg_" ++ name) orelse {
                // defensive only (the .dynsym check passed): its initializers ran, so it stays loaded, unused
                l.report.symbol = "tfg_" ++ name;
                l.report.pinned = true;
                return l.fail(.missing_symbol, 0, "dlsym found no tfg_{s} after the .dynsym check", .{name});
            };
            @field(f, name) = @as(T, @ptrCast(@alignCast(sym)));
        }
        l.fns = f;
        l.handle = handle;
        return true;
    }

    /// The 12 release names as defined global/weak default-visibility functions in the ELF .dynsym (section headers).
    fn elfSymbols(l: *Loader, fd: i32) bool {
        var eh: [64]u8 = undefined;
        if (!preadAll(fd, &eh, 0)) return l.fail(.not_elf, 0, "short ELF header", .{});
        const shoff = std.mem.readInt(u64, eh[0x28..0x30], .little);
        const shentsize = std.mem.readInt(u16, eh[0x3A..0x3C], .little);
        const shnum: usize = std.mem.readInt(u16, eh[0x3C..0x3E], .little);
        if (shentsize != 64 or shnum == 0 or shnum > 256) return l.fail(.not_elf, 0, "section headers {d} x {d}", .{ shnum, shentsize });
        var sh: [256 * 64]u8 = undefined;
        if (!preadAll(fd, sh[0 .. shnum * 64], shoff)) return l.fail(.not_elf, 0, "section headers unreadable", .{});
        var sym_off: u64 = 0;
        var sym_size: u64 = 0;
        var sym_ent: u64 = 0;
        var link: usize = shnum;
        for (0..shnum) |i| {
            const s = sh[i * 64 ..][0..64];
            if (std.mem.readInt(u32, s[4..8], .little) != 11) continue; // SHT_DYNSYM
            sym_off = std.mem.readInt(u64, s[0x18..0x20], .little);
            sym_size = std.mem.readInt(u64, s[0x20..0x28], .little);
            link = std.mem.readInt(u32, s[0x28..0x2C], .little);
            sym_ent = std.mem.readInt(u64, s[0x38..0x40], .little);
            break;
        }
        if (sym_ent != 24 or link >= shnum) return l.fail(.not_elf, 0, "no usable .dynsym", .{});
        const str = sh[link * 64 ..][0..64];
        const str_off = std.mem.readInt(u64, str[0x18..0x20], .little);
        const str_size = std.mem.readInt(u64, str[0x20..0x28], .little);
        const names = @typeInfo(Fns).@"struct".field_names;
        var have: [names.len]bool = @splat(false);
        var chunk: [24 * 256]u8 = undefined;
        const count = sym_size / 24;
        var at: u64 = 0;
        while (at < count) {
            const n: usize = @intCast(@min(256, count - at));
            if (!preadAll(fd, chunk[0 .. n * 24], sym_off + at * 24)) return l.fail(.not_elf, 0, ".dynsym unreadable", .{});
            for (0..n) |k| {
                const e = chunk[k * 24 ..][0..24];
                const st_name = std.mem.readInt(u32, e[0..4], .little);
                const bind = e[4] >> 4;
                const shndx = std.mem.readInt(u16, e[6..8], .little);
                if (shndx == 0 or e[4] & 0xf != 2 or (bind != 1 and bind != 2) or e[5] & 3 != 0 or st_name >= str_size) continue;
                var nm: [24]u8 = @splat(0xff);
                const len: usize = @intCast(@min(nm.len, str_size - st_name));
                if (!preadAll(fd, nm[0..len], str_off + st_name)) continue;
                inline for (names, 0..) |name, j| {
                    const full = "tfg_" ++ name;
                    if (len > full.len and std.mem.eql(u8, nm[0..full.len], full) and nm[full.len] == 0) have[j] = true;
                }
            }
            at += n;
        }
        inline for (names, 0..) |name, j| if (!have[j]) {
            l.report.symbol = "tfg_" ++ name;
            return l.fail(.missing_symbol, 0, "ELF .dynsym has no tfg_{s}; not loaded", .{name});
        };
        l.report.elf_symbols = true;
        return true;
    }

    fn parentSafe(l: *Loader, path: []const u8, euid: u32) bool {
        const cut = std.mem.lastIndexOfScalar(u8, path, '/').?;
        const dir = if (cut == 0) "/" else path[0..cut];
        var zbuf: [4096]u8 = undefined;
        @memcpy(zbuf[0..dir.len], dir);
        zbuf[dir.len] = 0;
        const rc = linux.open(@ptrCast(&zbuf), .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, 0);
        if (errnoOf(rc)) |e| return l.fail(.unsafe_parent, e, "parent open errno {d}", .{e});
        const fd: i32 = @intCast(rc);
        defer _ = linux.close(fd);
        var st: linux.Statx = undefined;
        if (errnoOf(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .TYPE = true, .MODE = true, .UID = true }, &st))) |e|
            return l.fail(.unsafe_parent, e, "parent statx errno {d}", .{e});
        if (st.uid != euid and st.uid != 0) return l.fail(.unsafe_parent, 0, "parent owner uid {d}", .{st.uid});
        if (st.mode & (linux.S.IWGRP | linux.S.IWOTH) != 0) return l.fail(.unsafe_parent, 0, "parent group/other writable (mode {o})", .{st.mode & 0o7777});
        return true;
    }
};

/// The process's loader (product use); tests make their own instances to exercise each refusal in one process.
pub var global: Loader = .{};

/// Absolute, no empty/"."/".." component, no NUL, shorter than PATH_MAX.
pub fn absoluteClean(path: []const u8) bool {
    if (path.len < 2 or path.len >= 4095 or path[0] != '/' or path[path.len - 1] == '/') return false;
    if (std.mem.indexOfScalar(u8, path, 0) != null) return false;
    var it = std.mem.splitScalar(u8, path[1..], '/');
    while (it.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return true;
}

pub fn errnoOf(rc: usize) ?u16 {
    const e = linux.errno(rc);
    return if (e == .SUCCESS) null else @intCast(@intFromEnum(e));
}

fn preadAll(fd: i32, buf: []u8, off: u64) bool {
    var got: usize = 0;
    while (got < buf.len) {
        const n = linux.pread(fd, buf[got..].ptr, buf.len - got, @intCast(off + got));
        if (errnoOf(n) != null or n == 0) return false;
        got += n;
    }
    return true;
}

pub fn hexLower(d: [32]u8) [64]u8 {
    const digits = "0123456789abcdef";
    var out: [64]u8 = undefined;
    for (d, 0..) |b, i| {
        out[2 * i] = digits[b >> 4];
        out[2 * i + 1] = digits[b & 15];
    }
    return out;
}
