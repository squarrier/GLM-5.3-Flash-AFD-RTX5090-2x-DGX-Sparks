//! The three-node exchange per MoE layer (qualification_12f4270a): the attention rank posts each layer's rows and routing to two expert ranks and takes their partials back, one pair over 0006's RoCE proxy to each expert.
//! Family-agnostic: messages are bytes; proto.Layout describes python TF-AFD 2.15's wire for any hidden size and slot count.
pub const proto = @import("proto.zig");
pub const watcher = @import("watcher.zig");
const node = @import("node.zig");
pub const Node = node.Node;
pub const Proxy = node.Proxy;
pub const Opener = node.Opener;
pub const Roce = node.Roce;
pub const Role = node.Role;
pub const Config = node.Config;
pub const Memory = node.Memory;
pub const Failure = node.Failure;
pub const Wait = node.Wait;
pub const tag_stop = node.tag_stop;
pub const gpu = @import("gpu.zig");
pub const Gpu = gpu.Gpu;
pub const Out = gpu.Out;
/// qualification_1fee6d78: TF_AFD_SETUP_MS, the handshake's window for one process (Config.setup_timeout_ms; unset 120,000 ms)
pub const setup = @import("setup.zig");

test {
    _ = proto;
    _ = watcher;
    _ = node;
    _ = gpu;
    _ = setup;
}
