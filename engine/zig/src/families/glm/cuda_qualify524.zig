//! AI-assisted integration: source-bound occupied-page checkpoints for one finite private524288x8 qualification, default-inert.
const std=@import("std");
const sha=std.crypto.hash.sha2.Sha256;
var frontier:usize=0;
const Row=struct {
 slot:u32,stream_ptr:usize,prompt_rows:usize,prompt_sha256:[32]u8,
 position:usize,mtp_position:usize,mtp_speculative_rows:usize,pending:usize,capacity:usize,
 pages:[]const u32,table:u64,conv:u64,recurrent:u64,projection:u64,mtp_pos:u64,cur_mask:u64,
 arena_ptr:u64,arena_bytes:usize,arena_sha256:[32]u8,history_sha256:[32]u8,history_bytes:usize,
};
/// Only 8completed -> 7aftercancel -> 8afterreplacement; no per-token capture or synthetic occupied rows.
pub fn checkpoint(self:anytype) !void {
 const v=std.c.getenv("TF_GLM_QUALIFY_524X8") orelse return;
 if(!std.mem.eql(u8,std.mem.span(v),"1") or frontier>=3)return;
 const wanted=if(frontier==1) @as(usize,7) else 8;
 if(self.lanes.count()!=wanted)return;
 var all=self.lanes.iterator();
 while(all.next()) |entry| {
  if(entry.key_ptr.*.prompt_len<477184 or entry.value_ptr.first==null)return;
 }
 const e=self.x.e;const pool=e.pages orelse return error.QualificationNeedsPages;
 const host=pool.host orelse return error.QualificationNeedsPrivateHostPages;
 try e.d.check(e.d.api.cuCtxSynchronize(),"qualification occupied context sync");
 try e.checkGathers();
 const scratch=try e.gpa.alloc(u8,1024*1024);defer e.gpa.free(scratch);
 var rows:[8]Row=undefined;var n:usize=0;var it=self.lanes.iterator();
 while(it.next()) |entry| {
  const stream=entry.key_ptr.*;const lane=entry.value_ptr;
  const seq=e.seqs.items[lane.slot] orelse return error.QualificationMissingState;
  const st=&seq.st;
  if(!pool.used[lane.slot] or st.pages.len!=pool.leases[lane.slot].ids.items.len)return error.QualificationLeaseMismatch;
  if(st.pos<stream.prompt_len)return error.QualificationNotWritten;
  var h=sha.init(.{});var history_bytes:usize=0;
  // The actual mandatory DSA writer chain committed State.pos; hash only those live host rows, never unwritten reservations.
  for(0..st.caches.len+1) |cache| {
   const live=if(cache==st.caches.len)st.mtp_len else st.pos;
   var logical:usize=0;
   for(st.pages) |id| {
    if(logical>=live)break;
    const take:usize=@min(@as(usize,512),live-logical);
    const at=@as(usize,id)*pool.page+cache*(512*1024+128*256);
    const bytes=take*1024;
    if(at+bytes>host.bytes.len)return error.QualificationHostBounds;
    h.update(host.bytes[at..][0..bytes]);history_bytes+=bytes;logical+=take;
   }
   if(logical!=live)return error.QualificationShortLease;
  }
  var a=sha.init(.{});var at:usize=0;
  while(at<st.arena.buf.len) {
   const take=@min(scratch.len,st.arena.buf.len-at);
   try e.d.check(e.d.api.cuMemcpyDtoH_v2(scratch.ptr,st.arena.buf.ptr+at,take),"qualification state readback");
   a.update(scratch[0..take]);at+=take;
  }
  var prompt_hash:[32]u8=undefined;sha.hash(std.mem.sliceAsBytes(stream.prompt()),&prompt_hash,.{});
  var cur:u64=0;for(st.cur,0..) |bit,i| {if(i>=64)return error.QualificationStateShape;cur|=@as(u64,bit)<<@intCast(i);}
  rows[n]=.{.slot=lane.slot,.stream_ptr=@intFromPtr(stream),.prompt_rows=stream.prompt_len,.prompt_sha256=prompt_hash,
   .position=st.pos,.mtp_position=st.mtp_len,.mtp_speculative_rows=seq.in_head,.pending=st.pending,.capacity=st.capacity,
   .pages=st.pages,.table=st.table,.conv=st.conv,.recurrent=st.rec,.projection=st.proj,.mtp_pos=st.mtp_pos_dev,.cur_mask=cur,
   .arena_ptr=st.arena.buf.ptr,.arena_bytes=st.arena.buf.len,.arena_sha256=a.finalResult(),.history_sha256=h.finalResult(),.history_bytes=history_bytes};
  n+=1;
 }
 const text=try std.json.Stringify.valueAlloc(e.gpa,.{.phase=frontier,.completed=n,.pool=pool.pool.census(),.host_bytes=host.bytes.len,.page_stride=pool.page,.rows=rows[0..n],.scope="synchronized_committed_host_rows_and_complete_sequence_arenas_not_numerical_oracle"},.{});
 defer e.gpa.free(text);
 std.debug.print("OCCUPIED524 {s}\n",.{text});frontier+=1;
}
