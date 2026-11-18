//! The actual native Loader/Checkpoint/MoE path with synthetic shard bytes and recording uploads.
const std=@import("std");
const cuda=@import("cuda");
const core=@import("core");
const weights=@import("cuda_weights.zig");
const shard=@import("cuda_shard.zig");
const t=std.testing;
const Upload=struct {
    var next: u64=0x100000;
    var allocations: usize=0;var releases: usize=0;var copies:usize=0;var cancel:?*std.atomic.Value(bool)=null;
    fn alloc(p: *u64,n: usize) callconv(.c) c_int { p.*=next;next+=std.mem.alignForward(u64,n,256);allocations+=1;return 0; }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int { return 0; }
    fn free(_: u64) callconv(.c) c_int { releases+=1;return 0; }
    fn copy(_: u64,_: ?*const anyopaque,_: usize) callconv(.c) c_int { copies+=1;if(cancel)|flag|flag.store(true,.release);return 0; }
};
const Fixture=struct {
    g: std.mem.Allocator,
    names: core.safetensors.Header=.empty,
    bytes: std.ArrayList(u8)=.empty,
    fn add(f: *Fixture,comptime suffix: []const u8,args: anytype,dtype: core.safetensors.DType,shape: []const usize) !void {
        const name=try std.fmt.allocPrint(f.g,"model.language_model."++suffix,args);
        var e: core.safetensors.Entry=.{.dtype=dtype,.rank=@intCast(shape.len),.shape=@splat(1),.begin=f.bytes.items.len,.end=0};
        var n=dtype.size();for(shape,0..) |dim,i| {e.shape[i]=dim;n*=dim;}
        try f.bytes.appendNTimes(f.g,0,n);e.end=f.bytes.items.len;
        for(shape,0..) |dim,i| e.shape[i]=dim;
        if(std.mem.endsWith(u8,name,".mcg")) std.mem.writeInt(u32,f.bytes.items[e.begin..][0..4],core.exl3_load.markers[0].value,.little);
        try f.names.put(f.g,name,e);
    }
    fn moe(f: *Fixture,i: u32) !void {
        try f.add("layers.{d}.mlp.gate.weight",.{i},.bf16,&.{1,512});
        try f.add("layers.{d}.mlp.gate.e_score_correction_bias",.{i},.f32,&.{1});
        inline for(.{"gate_proj","up_proj","down_proj"},0..) |proj,m| {
            const k: usize=if(m<2) 512 else 256;const n: usize=if(m<2) 256 else 512;
            try f.add("layers.{d}.mlp.experts.0."++proj++".trellis",.{i},.i16,&.{k/16,n/16,64});
            try f.add("layers.{d}.mlp.experts.0."++proj++".mcg",.{i},.i32,&.{1});
            try f.add("layers.{d}.mlp.experts.0."++proj++".suh",.{i},.f16,&.{k});
            try f.add("layers.{d}.mlp.experts.0."++proj++".svh",.{i},.f16,&.{n});
            try f.add("layers.{d}.mlp.shared_experts."++proj++".weight",.{i},.bf16,if(m<2) &.{64,512} else &.{512,64});
        }
    }
};

pub fn expert(comptime Loader: type) !void {
    var arena=std.heap.ArenaAllocator.init(t.allocator);defer arena.deinit();const a=arena.allocator();
    var fx=Fixture{.g=a};try fx.moe(0);try fx.moe(1);
    const memory=try a.alignedAlloc(u8,.fromByteUnits(4096),fx.bytes.items.len);@memcpy(memory,fx.bytes.items);
    var file: core.safetensors.File=undefined;file.names=fx.names;file.data=0;file.map.memory=memory;file.map.section=null;file.file=.{.handle=-1,.flags=.{.nonblocking=false}};
    var ck: core.Checkpoint=.{.gpa=a,.io=undefined};try ck.files.append(a,file);
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=Upload.alloc;d.api.cuMemFree_v2=Upload.free;d.api.cuMemcpyHtoD_v2=Upload.copy;d.api.cuStreamSynchronize=Upload.sync;
    const c=@import("config.zig").Config{.hidden=512,.moe_inter=256,.shared_inter=64,.experts=1,.exl3=true,.layers=1,.dense_layers=0,.own=.{0,1}};
    if(comptime @hasDecl(weights,"loadExpertLayers")) {
        for([_]u1{0,1}) |half| {
            var w=try weights.loadExpertLayers(a,&d,.{.d=&d,.handle=null},&ck,c,half,&.{0},true);
            defer w.deinit();
            try t.expectEqual(@as(usize,2),w.layers.len);
            try t.expectEqual(@as(usize,128),w.cut.len());
            try t.expectEqual(@as(u32,1),w.layers[1].index);
            for(w.named.items) |n| try t.expect(std.mem.indexOf(u8,n.name,".exl3.")!=null);
        }
    } else {
        var w: weights.Weights=.{.gpa=a,.config=c,.share=try shard.Share.of(&c,0,1)};defer w.deinit();
        var L: Loader=.{.gpa=a,.d=&d,.stream=.{.d=&d,.handle=null},.pack_dense=undefined,.tools=undefined,.w=&w,.ck=&ck};defer L.host.deinit(a);
        _=try L.moeLayer(0);_=try L.moeLayer(1);
    }
    var reads=ck.used.keyIterator();var forbidden: usize=0;
    while(reads.next()) |name| if(std.mem.indexOf(u8,name.*,".experts.")==null) {forbidden+=1;};
    try t.expectEqual(@as(usize,0),forbidden);
}

pub fn attention(comptime Loader: type) !void {
    var arena=std.heap.ArenaAllocator.init(t.allocator);defer arena.deinit();const a=arena.allocator();
    var fx=Fixture{.g=a};try fx.moe(0);try fx.moe(1); // native MTP layer index is config.layers
    const memory=try a.alignedAlloc(u8,.fromByteUnits(4096),fx.bytes.items.len);@memcpy(memory,fx.bytes.items);
    var file: core.safetensors.File=undefined;file.names=fx.names;file.data=0;file.map.memory=memory;file.map.section=null;file.file=.{.handle=-1,.flags=.{.nonblocking=false}};
    var ck: core.Checkpoint=.{.gpa=a,.io=undefined};try ck.files.append(a,file);
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=Upload.alloc;d.api.cuMemFree_v2=Upload.free;d.api.cuMemcpyHtoD_v2=Upload.copy;d.api.cuStreamSynchronize=Upload.sync;
    const c=@import("config.zig").Config{.hidden=512,.moe_inter=256,.shared_inter=64,.experts=1,.exl3=true,.layers=1};
    var w: weights.Weights=.{.gpa=a,.config=c,.share=try shard.Share.of(&c,0,1)};
    if(comptime @hasField(weights.Weights,"role")) w.role=.attention;
    defer w.deinit();
    var L: Loader=.{.gpa=a,.d=&d,.stream=.{.d=&d,.handle=null},.pack_dense=undefined,.tools=undefined,.w=&w,.ck=&ck};
    defer L.host.deinit(a);
    _=try L.moeLayer(0);_=try L.moeLayer(1);
    var reads=ck.used.keyIterator();var forbidden: usize=0;
    while(reads.next()) |name| if(std.mem.indexOf(u8,name.*,".experts.")!=null) {forbidden+=1;};
    try t.expectEqual(@as(usize,0),forbidden);
    for(w.named.items) |n| try t.expect(std.mem.indexOf(u8,n.name,".exl3.")==null);
}

// Helios: actual scaled loader boundary for the real hardware's late TERM and stale setup budget.
pub fn controlled(cancel_after_copy:bool) !void {
    const H=@import("cuda_hybrid_load.zig");var arena=std.heap.ArenaAllocator.init(t.allocator);defer arena.deinit();const a=arena.allocator();
    var fx=Fixture{.g=a};try fx.moe(0);try fx.moe(1);const memory=try a.alignedAlloc(u8,.fromByteUnits(4096),fx.bytes.items.len);@memcpy(memory,fx.bytes.items);
    var file:core.safetensors.File=undefined;file.names=fx.names;file.data=0;file.map.memory=memory;file.map.section=null;file.file=.{.handle=-1,.flags=.{.nonblocking=false}};var ck:core.Checkpoint=.{.gpa=a,.io=undefined};try ck.files.append(a,file);
    var d:cuda.Driver=undefined;d.api.cuMemAlloc_v2=Upload.alloc;d.api.cuMemFree_v2=Upload.free;d.api.cuMemcpyHtoD_v2=Upload.copy;d.api.cuStreamSynchronize=Upload.sync;
    const c=@import("config.zig").Config{.hidden=512,.moe_inter=256,.shared_inter=64,.experts=1,.exl3=true,.layers=1,.dense_layers=0,.own=.{0,1}};
    var stop=std.atomic.Value(bool).init(false);Upload.allocations=0;Upload.releases=0;Upload.copies=0;Upload.cancel=if(cancel_after_copy)&stop else null;defer Upload.cancel=null;
    const result=if(comptime @hasDecl(H,"loadExpertLayersControlled")) H.loadExpertLayersControlled(a,&d,.{.d=&d,.handle=null},&ck,c,0,&.{0},true,.{.stopping=&stop,.until=if(cancel_after_copy)null else 0}) else H.loadExpertLayers(a,&d,.{.d=&d,.handle=null},&ck,c,0,&.{0},true);
    if(result)|value| {var owned=value;owned.deinit();return error.TestExpectedError;} else |err|try t.expectEqual(if(cancel_after_copy)error.Cancelled else error.Timeout,err);
    try t.expectEqual(@as(usize,if(cancel_after_copy)1 else 0),Upload.copies);try t.expectEqual(Upload.allocations,Upload.releases);
    if(!cancel_after_copy)try t.expectEqual(@as(usize,0),ck.used.count());
}


// Actual mapped checkpoint traversal; CUDA calls only record complete upload bytes.
const Reclaim=struct {
    var first:*core.safetensors.File=undefined;
    var calls:usize=0;var frees:usize=0;var next:u64=0x100000;
    var previous_layer_still_mapped:bool=false;var probe_failed:bool=false;
    var hash:std.crypto.hash.sha2.Sha256=undefined;
    fn rss(file:*const core.safetensors.File)!usize {
        const input=try std.Io.Dir.cwd().openFile(t.io,"/proc/self/smaps",.{});defer input.close(t.io);var buffer:[4096]u8=undefined;var reader=input.readerStreaming(t.io,&buffer);const text=try reader.interface.allocRemaining(t.allocator,.limited(8<<20));defer t.allocator.free(text);
        var lines=std.mem.splitScalar(u8,text,'\n');var selected=false;
        const address=@intFromPtr(file.map.memory.ptr);
        while(lines.next())|line| {
            if(std.mem.indexOfScalar(u8,line,'-'))|dash| {
                if(std.mem.indexOfScalar(u8,line,' '))|space|if(dash<space) {
                    const lo=std.fmt.parseInt(usize,line[0..dash],16) catch continue;
                    const hi=std.fmt.parseInt(usize,line[dash+1..space],16) catch continue;
                    selected=address>=lo and address<hi;
                };
            }
            if(selected and std.mem.startsWith(u8,line,"Rss:"))return std.fmt.parseInt(usize,std.mem.trim(u8,line[4..line.len-2]," \t"),10);
        }
        return error.MappingMissing;
    }
    fn alloc(out:*u64,n:usize)callconv(.c)c_int {
        calls+=1;
        if(calls==14) {const n_rss=rss(first) catch |err| blk:{probe_failed=true;std.debug.print("checkpoint RSS probe failed: {t}\n",.{err});break :blk 0;};previous_layer_still_mapped=n_rss!=0;}
        out.*=next;next+=std.mem.alignForward(u64,n,256);return 0;
    }
    fn free(_:u64)callconv(.c)c_int {frees+=1;return 0;}
    fn sync(_:cuda.abi.Stream)callconv(.c)c_int {return 0;}
    fn take(h:*std.crypto.hash.sha2.Sha256,bytes:[]const u8)void {
        var n:[8]u8=undefined;std.mem.writeInt(u64,&n,bytes.len,.little);h.update(&n);h.update(bytes);
    }
    fn copy(_:u64,ptr:?*const anyopaque,n:usize)callconv(.c)c_int {take(&hash,@as([*]const u8,@ptrCast(ptr.?))[0..n]);return 0;}
};
fn image(f:*Fixture)![]u8 {
    var header:std.ArrayList(u8)=.empty;defer header.deinit(f.g);try header.append(f.g,'{');
    for(f.names.keys(),f.names.values(),0..)|name,e,i| {
        if(i>0)try header.append(f.g,',');
        const key=try std.json.Stringify.valueAlloc(f.g,name,.{});defer f.g.free(key);try header.appendSlice(f.g,key);try header.append(f.g,':');
        const dtype=switch(e.dtype){.bf16=>"BF16",.f32=>"F32",.i16=>"I16",.i32=>"I32",.f16=>"F16",else=>unreachable};
        const value=try std.json.Stringify.valueAlloc(f.g,.{.dtype=dtype,.shape=e.shape[0..e.rank],.data_offsets=[_]usize{e.begin,e.end}},.{});defer f.g.free(value);try header.appendSlice(f.g,value);
    }
    try header.append(f.g,'}');const bytes=try f.g.alloc(u8,8+header.items.len+f.bytes.items.len);
    std.mem.writeInt(u64,bytes[0..8],header.items.len,.little);@memcpy(bytes[8..][0..header.items.len],header.items);@memcpy(bytes[8+header.items.len..],f.bytes.items);return bytes;
}
fn expectedUpload(h:*std.crypto.hash.sha2.Sha256,address:*u64,bytes:[]const u8)u64 {
    const start=address.*;address.*+=std.mem.alignForward(u64,bytes.len,256);Reclaim.take(h,bytes);return start;
}
fn expectedLayer(f:*Fixture,i:u32,half:u1,h:*std.crypto.hash.sha2.Sha256,address:*u64)!void {
    var scratch:std.ArrayList(u8)=.empty;defer scratch.deinit(f.g);var name:[192]u8=undefined;
    inline for(.{"gate_proj","up_proj","down_proj"},0..)|proj,m| {
        const k:usize=if(m<2)512 else 256;const n:usize=if(m<2)256 else 512;
        const e=f.names.get(try std.fmt.bufPrint(&name,"model.language_model.layers.{d}.mlp.experts.0."++proj++".trellis",.{i})).?;
        const source=f.bytes.items[e.begin..e.end];scratch.clearRetainingCapacity();
        if(m<2) {for(0..k/16)|row|try scratch.appendSlice(f.g,source[row*(n/16)*128+@as(usize,half)*8*128..][0..8*128]);}
        else try scratch.appendSlice(f.g,source[@as(usize,half)*8*(n/16)*128..][0..8*(n/16)*128]);
        const ptr=expectedUpload(h,address,scratch.items);var pointer:[8]u8=undefined;std.mem.writeInt(u64,&pointer,ptr,.little);_=expectedUpload(h,address,&pointer);
        inline for(.{"suh","svh"},0..)|scale,s| {
            const entry=f.names.get(try std.fmt.bufPrint(&name,"model.language_model.layers.{d}.mlp.experts.0."++proj++"."++scale,.{i})).?;
            const values=f.bytes.items[entry.begin..entry.end];const sliced=(s==0 and m==2)or(s==1 and m<2);
            _=expectedUpload(h,address,if(sliced)values[@as(usize,half)*256..][0..256]else values);
        }
    }
    const bits=[_]u8{8,0,0,0};_=expectedUpload(h,address,&bits);
}
pub fn reclamation()!void {
    if(@import("builtin").os.tag!=.linux)return error.SkipZigTest;
    var arena=std.heap.ArenaAllocator.init(t.allocator);defer arena.deinit();const a=arena.allocator();
    const previous=try std.Io.Dir.cwd().openDir(t.io,".",.{});defer previous.close(t.io);try t.expectEqual(@as(c_int,0),std.c.chdir(std.c.getenv("TMPDIR") orelse "."));defer _=std.c.fchdir(previous.handle);
    var tmp=t.tmpDir(.{});defer tmp.cleanup();var fixtures=[_]Fixture{.{.g=a},.{.g=a}};var originals:[2][]u8=undefined;
    for(&fixtures,0..)|*f,i| {try f.moe(@intCast(i));for(f.names.keys(),f.names.values())|name,e| {
        if(!std.mem.endsWith(u8,name,".mcg")){for(f.bytes.items[e.begin..e.end],0..)|*byte,j|{byte.*=@truncate(j*37+i*53+e.begin);}}
    } originals[i]=try image(f);const name=try std.fmt.allocPrint(a,"model-{d}.safetensors",.{i});try tmp.dir.writeFile(t.io,.{.sub_path=name,.data=originals[i]});}
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);var ck=try core.Checkpoint.openModelPrefix(a,t.io,dir,"model.language_model.");defer ck.close();
    var d:cuda.Driver=undefined;d.api.cuMemAlloc_v2=Reclaim.alloc;d.api.cuMemFree_v2=Reclaim.free;d.api.cuMemcpyHtoD_v2=Reclaim.copy;d.api.cuStreamSynchronize=Reclaim.sync;
    const c=@import("config.zig").Config{.hidden=512,.moe_inter=256,.shared_inter=64,.experts=1,.exl3=true,.layers=1,.dense_layers=0,.own=.{0,1}};
    for([_]u1{0,1})|half| {
        ck.release();Reclaim.first=&ck.files.items[0];Reclaim.calls=0;Reclaim.frees=0;Reclaim.next=0x100000;Reclaim.previous_layer_still_mapped=false;Reclaim.probe_failed=false;Reclaim.hash=.init(.{});
        var oracle=std.crypto.hash.sha2.Sha256.init(.{});var address:u64=0x100000;for(&fixtures,0..)|*f,i|try expectedLayer(f,@intCast(i),half,&oracle,&address);
        var owned=try @import("cuda_hybrid_load.zig").loadExpertLayers(a,&d,.{.d=&d,.handle=null},&ck,c,half,&.{0},true);
        const first_rss=try Reclaim.rss(&ck.files.items[0]);const mtp_rss=try Reclaim.rss(&ck.files.items[1]);owned.deinit();
        try t.expectEqual(@as(usize,26),Reclaim.calls);try t.expectEqual(Reclaim.calls,Reclaim.frees);try t.expectEqual(oracle.finalResult(),Reclaim.hash.finalResult());
        try t.expect(!Reclaim.probe_failed);try t.expect(!Reclaim.previous_layer_still_mapped);try t.expectEqual(@as(usize,0),first_rss);try t.expectEqual(@as(usize,0),mtp_rss);
    }
    for(originals,0..)|bytes,i| {const name=try std.fmt.allocPrint(a,"model-{d}.safetensors",.{i});const actual=try tmp.dir.readFileAlloc(t.io,name,a,.limited(2<<20));try t.expectEqualSlices(u8,bytes,actual);}
}

// Helios: real attention Loader cancellation after its first synchronous recording upload.
pub fn attentionControlled(comptime Loader:type)!void {
    var arena=std.heap.ArenaAllocator.init(t.allocator);defer arena.deinit();const a=arena.allocator();var fx=Fixture{.g=a};try fx.moe(0);try fx.moe(1);
    const memory=try a.alignedAlloc(u8,.fromByteUnits(4096),fx.bytes.items.len);@memcpy(memory,fx.bytes.items);
    var file:core.safetensors.File=undefined;file.names=fx.names;file.data=0;file.map.memory=memory;file.map.section=null;file.file=.{.handle=-1,.flags=.{.nonblocking=false}};var ck:core.Checkpoint=.{.gpa=a,.io=undefined};try ck.files.append(a,file);
    var d:cuda.Driver=undefined;d.api.cuMemAlloc_v2=Upload.alloc;d.api.cuMemFree_v2=Upload.free;d.api.cuMemcpyHtoD_v2=Upload.copy;d.api.cuStreamSynchronize=Upload.sync;
    const c=@import("config.zig").Config{.hidden=512,.moe_inter=256,.shared_inter=64,.experts=1,.exl3=true,.layers=1};
    var w:weights.Weights=.{.gpa=a,.config=c,.share=try shard.Share.of(&c,0,1),.role=.attention};var closed=false;defer if(!closed)w.deinit();
    var L:Loader=.{.gpa=a,.d=&d,.stream=.{.d=&d,.handle=null},.pack_dense=undefined,.tools=undefined,.w=&w,.ck=&ck};defer L.host.deinit(a);
    var stop=std.atomic.Value(bool).init(false);Upload.allocations=0;Upload.releases=0;Upload.copies=0;Upload.cancel=&stop;defer Upload.cancel=null;
    if(comptime @hasField(Loader,"control"))L.control=.{.stopping=&stop};
    try t.expectError(error.Cancelled,L.moeLayer(0));
    w.deinit();closed=true;try t.expect(stop.load(.acquire));try t.expectEqual(@as(usize,1),Upload.copies);try t.expectEqual(Upload.allocations,Upload.releases);
}
