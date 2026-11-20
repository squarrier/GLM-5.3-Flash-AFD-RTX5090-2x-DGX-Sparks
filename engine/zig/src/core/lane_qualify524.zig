//! AI-assisted integration: finite trial-only pause of completed near-full prompts; default-inert, native Stream lifetime retained.
const std=@import("std");
pub fn selected() bool {
 const value=std.c.getenv("TF_GLM_QUALIFY_524X8") orelse return false;
 return std.mem.eql(u8,std.mem.span(value),"1");
}
pub const Gate=struct {
 enabled:bool=false,
 seen:[9]u64=undefined,
 seen_n:usize=0,
 released:bool=false,
 expired:bool=false,
 until:?i96=null,
 pub fn fromEnv() Gate {return .{.enabled=selected()};}
 fn owns(g:*const Gate,seq:u64) bool {
  for(g.seen[0..g.seen_n]) |id| if(id==seq)return true;
  return false;
 }
 fn release(g:*Gate,jobs:anytype) void {
  for(jobs) |j| if(j.started and g.owns(j.seq)) {j.stream.paused=false;};
  g.released=true;
 }
 /// The engine thread calls after fill.run returns, never from the prefilled callback or an HTTP thread.
 pub fn visit(g:*Gate,jobs:anytype,now:i96) bool {
  if(!g.enabled or g.released)return false;
  if(g.until) |end| if(now>=end) {
   g.expired=true;g.release(jobs);
   std.debug.print("QUALIFY524 {{\"phase\":\"hold_expired\",\"seen\":{d}}}\n",.{g.seen_n});
   return false;
  };
  for(jobs) |j| {
   if(!j.started or j.prefilled==null or j.stream.finished or j.stream.prompt_len<477184)continue;
   if(!g.owns(j.seq)) {
    if(g.seen_n>=g.seen.len) {g.expired=true;g.release(jobs);return false;}
    g.seen[g.seen_n]=j.seq;g.seen_n+=1;
    if(g.until==null)g.until=now+5400*std.time.ns_per_s;
    std.debug.print("QUALIFY524 {{\"phase\":\"held\",\"seq\":{d},\"stream_ptr\":{d},\"seen\":{d},\"prompt_rows\":{d}}}\n",.{j.seq,@intFromPtr(&j.stream),g.seen_n,j.stream.prompt_len});
   }
   j.stream.paused=true;
  }
  if(g.seen_n==9) {
   g.release(jobs);
   std.debug.print("QUALIFY524 {{\"phase\":\"replacement_prefilled_release\",\"seen\":9}}\n",.{});
   return false;
  }
  var waiting=false;
  for(jobs) |j| {
   if(!j.started or j.stream.finished)continue;
   if(!j.stream.paused)return false;
   waiting=true;
  }
  return waiting;
 }
};
