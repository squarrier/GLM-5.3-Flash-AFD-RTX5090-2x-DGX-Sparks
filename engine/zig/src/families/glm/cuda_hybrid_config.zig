//! Portable explicit role/command contract, shared by the server initialization and the actual expert executable.
const std=@import("std");
const ranks=@import("cuda_ranks.zig");
const transport=@import("cuda_hybrid_transport.zig");
pub const Role=enum {attention,expert0,expert1};
pub const Config=struct {
    revision: u32,role: Role,session: []const u8,model_sha256: []const u8,
    master: [2][]const u8,ports: [2]u16,traffic_class: u8,verbs_device: []const u8,roce_image: []const u8,
    setup_ms: u32,transaction_ms: u32,close_ms: u32,
    served_context: u32,resident_users: u32,mtp: bool,settings: ranks.Settings,
    pub fn half(c: Config) !u1 {return switch(c.role) {.expert0=>0,.expert1=>1,.attention=>error.BadHybridRole};}
    pub fn identity(c: Config,a: std.mem.Allocator) !transport.Identity {
        if(c.session.len!=32 or c.model_sha256.len!=64) return error.BadHybridIdentity;
        var session_bytes: [16]u8=undefined;_=std.fmt.hexToBytes(&session_bytes,c.session) catch return error.BadHybridIdentity;
        const session=std.mem.readInt(u128,&session_bytes,.big);if(session==0) return error.BadHybridIdentity;
        var model: [32]u8=undefined;_=std.fmt.hexToBytes(&model,c.model_sha256) catch return error.BadHybridIdentity;
        if(std.mem.allEqual(u8,&model,0)) return error.BadHybridIdentity;
        return .{.session=session,.model=model,.settings=try digest(a,c.settings,c.served_context,c.mtp,c.traffic_class,c.setup_ms,c.transaction_ms,c.close_ms),.transaction_timeout_ns=@as(u64,c.transaction_ms)*std.time.ns_per_ms};
    }
    pub fn attention(c: Config,a: std.mem.Allocator,world: usize,context: usize,mtp: bool,set: ranks.Settings) !void {
        if(c.role!=.attention or world!=1) return error.BadHybridRole;
        if(context!=c.served_context or mtp!=c.mtp) return error.BadHybridContract;
        const declared=try c.identity(a);const actual=try digest(a,set,context,mtp,c.traffic_class,c.setup_ms,c.transaction_ms,c.close_ms);
        if(!std.mem.eql(u8,&declared.settings,&actual)) return error.BadHybridContract;
    }
};
fn digest(a: std.mem.Allocator,set: ranks.Settings,context: usize,mtp: bool,traffic_class: u8,setup_ms: u32,transaction_ms: u32,close_ms: u32) ![32]u8 {
    const text=try std.json.Stringify.valueAlloc(a,.{.abi="WIRE3-ordered-FP32-half0-half1-shared-budget1",.settings=set,.served_context=context,.mtp=mtp,.traffic_class=traffic_class,.setup_ms=setup_ms,.transaction_ms=transaction_ms,.close_ms=close_ms},.{});defer a.free(text);
    return ranks.hashBytes(text);
}
pub fn parse(a: std.mem.Allocator,text: []const u8) !std.json.Parsed(Config) {
    if(text.len>65536) return error.BadHybridContract;
    var p=try std.json.parseFromSlice(Config,a,text,.{.allocate=.alloc_always,.ignore_unknown_fields=false,.duplicate_field_behavior=.@"error"});errdefer p.deinit();
    const c=p.value;
    if(c.revision!=3 or c.served_context==0 or c.resident_users==0 or c.ports[0]==0 or c.ports[1]==0 or c.ports[0]==c.ports[1] or c.verbs_device.len==0 or c.verbs_device.len>255 or c.roce_image.len==0) return error.BadHybridContract;
    if(c.ports[0]>65528 or c.ports[1]>65528 or c.ports[0]+7==c.ports[1] or c.ports[1]+7==c.ports[0]) return error.BadHybridContract;
    for(c.master) |host| if(host.len==0 or host.len>255) return error.BadHybridContract;
    inline for(.{c.setup_ms,c.transaction_ms,c.close_ms}) |ms| if(ms==0 or ms>3_600_000) return error.BadHybridContract;
    _=try c.identity(a);return p;
}

pub const Cli=struct {
    config: []const u8,checkpoint: []const u8,device: c_int,check: bool=false,
    pub fn parse(args: []const []const u8) !Cli {
        var config: ?[]const u8=null;var checkpoint: ?[]const u8=null;var device: ?c_int=null;var check=false;var i: usize=1;
        while(i<args.len) : (i+=1) {
            const key=args[i];
            if(std.mem.eql(u8,key,"--check")) {if(check) return error.BadExpertArgs;check=true;continue;}
            if(i+1>=args.len) return error.BadExpertArgs;const value=args[i+1];i+=1;if(value.len==0) return error.BadExpertArgs;
            if(std.mem.eql(u8,key,"--config") and config==null) config=value
            else if(std.mem.eql(u8,key,"--checkpoint") and checkpoint==null) checkpoint=value
            else if(std.mem.eql(u8,key,"--device") and device==null) {device=std.fmt.parseInt(c_int,value,10) catch return error.BadExpertArgs;if(device.?<0) return error.BadExpertArgs;}
            else return error.BadExpertArgs;
        }
        return .{.config=config orelse return error.BadExpertArgs,.checkpoint=checkpoint orelse return error.BadExpertArgs,.device=device orelse return error.BadExpertArgs,.check=check};
    }
};
