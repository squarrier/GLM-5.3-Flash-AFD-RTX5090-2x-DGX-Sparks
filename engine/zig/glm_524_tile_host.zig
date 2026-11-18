//! Helios: actual selected native sparse/dense consumers with a recording-only driver; no numerical qualification.
const std=@import("std");
const fixture=@import("glm_kv_host_hybrid_host.zig");
extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int)c_int;
extern "c" fn unsetenv([*:0]const u8)c_int;
test "qualification524 tiled host sparse consumer covers every row and retains original prompt pointers" {
 try std.testing.expect(std.c.getenv("TF_GLM_ATTENTION_TILE_ROWS")==null);
 try std.testing.expectEqual(@as(c_int,0),setenv("TF_GLM_ATTENTION_TILE_ROWS","16",1));defer _=unsetenv("TF_GLM_ATTENTION_TILE_ROWS");
 for([_]usize{33,65,127,512,2048,2066}) |rows| {
  const r=try fixture.tile524(rows,16,30000);const last=(rows-1)/16*16;
  try std.testing.expectEqual((rows+15)/16,r.calls);try std.testing.expectEqual(r.calls,r.copies);
  try std.testing.expectEqual(r.base_tokens+last*2051*4,r.tokens);
  try std.testing.expectEqual(r.base_counts+last*4,r.counts);
  try std.testing.expectEqual(r.base_qa+last*16*512*2,r.qa);
 }
}
test "qualification524 decode verify32 minimum stays untiled and dense temporary blocks are bounded" {
 try std.testing.expect(std.c.getenv("TF_GLM_ATTENTION_TILE_ROWS")==null);
 try std.testing.expectEqual(@as(c_int,0),setenv("TF_GLM_ATTENTION_TILE_ROWS","16",1));defer _=unsetenv("TF_GLM_ATTENTION_TILE_ROWS");
 for([_]usize{1,8,16,17,32}) |rows| {const r=try fixture.tile524(rows,32,30000);try std.testing.expectEqual(@as(usize,1),r.calls);try std.testing.expectEqual(r.base_qa,r.qa);}
 const dense=try fixture.tile524(65,16,0);try std.testing.expectEqual(@as(usize,5),dense.calls);try std.testing.expectEqual(@as(usize,0),dense.copies);
}
