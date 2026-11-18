//! Qualification-only pause gate tests; no CUDA, model or network.
const std=@import("std");
const Gate=@import("lane_qualify524.zig").Gate;
const Stream=struct {paused:bool=false,finished:bool=false,prompt_len:usize=522240};
const Request=struct {prompt:[]const u32=&.{}};
const Job=struct {seq:u64,started:bool=true,prefilled:?i96=1,stream:Stream=.{}};
test "qualification524 holds actual completed jobs through cancel and replacement" {
 var gate:Gate=.{.enabled=true};var jobs:[8]Job=undefined;var ptrs:[8]*Job=undefined;
 for(&jobs,&ptrs,0..) |*j,*p,i| {j.*=.{.seq=i};p.*=j;}
 try std.testing.expect(gate.visit(&ptrs,100));
 for(jobs) |j| try std.testing.expect(j.stream.paused);
 try std.testing.expectEqual(@as(usize,8),gate.seen_n);
 // One actual cancellation removes its job; the other seven stay paused and do not duplicate enrollment.
 try std.testing.expect(gate.visit(ptrs[0..7],200));
 try std.testing.expectEqual(@as(usize,8),gate.seen_n);
 var replacement:Job=.{.seq=99};ptrs[7]=&replacement;
 try std.testing.expect(!gate.visit(&ptrs,300));
 for(ptrs) |j| try std.testing.expect(!j.stream.paused);
 try std.testing.expect(gate.released and !gate.expired);
}
