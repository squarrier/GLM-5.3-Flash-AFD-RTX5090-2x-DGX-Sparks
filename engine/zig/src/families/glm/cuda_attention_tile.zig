//! AI-assisted integration: optional temporary-attention tiling, without changing prompt/KDA/verify geometry or numeric kernels.
const std=@import("std");
pub fn parse(text:?[]const u8) !?usize {
 const t=text orelse return null;
 const n=std.fmt.parseInt(usize,t,10) catch return error.BadAttentionTile;
 return switch(n){16,32,64,128=>n,else=>error.BadAttentionTile};
}
pub fn rows() !?usize {
 const text=std.c.getenv("TF_GLM_ATTENTION_TILE_ROWS") orelse return null;
 return parse(std.mem.span(text));
}
pub fn prefillRows(normal:usize) !usize {return if(try rows()) |r| @min(normal,r) else normal;}
pub fn sparseRows(normal:usize) !usize {
 // Preserve the complete32-row verify/decode minimum, including shared-MTP windows.
 return if(normal>32) if(try rows()) |r| @min(normal,r) else normal else normal;
}
pub fn gatherCapacity(capacity:usize) !usize {
 // A tiled prompt and an UNTILED32-row shared verify are both consumers of the one gather.
 return if(try rows()) |r| @min(capacity,@max(r,32)*2051) else capacity;
}
test "qualification524 attention tile is explicit bounded and default-inert" {
 try std.testing.expectEqual(@as(?usize,null),try parse(null));
 inline for(.{16,32,64,128}) |n| try std.testing.expectEqual(@as(?usize,n),try parse(std.fmt.comptimePrint("{d}",.{n})));
 inline for(.{"0","1","3","2048","nan",""}) |t| try std.testing.expectError(error.BadAttentionTile,parse(t));
}
