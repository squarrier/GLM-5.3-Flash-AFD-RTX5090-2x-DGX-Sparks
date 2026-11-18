// SPDX-License-Identifier: Apache-2.0
//! Optional process-exclusive buffer ownership witness; no I/O, allocation or device calls.
//! Default inert. A copied witness follows its actual backing across threads/scopes.
const std=@import("std");
pub const Kind=enum{device,pinned,hooked_range};
pub const Ownership=struct{
    schema:u32=1,role:[]const u8,incarnation:u128,live_device:u64,live_pinned:u64,retained_device:u64,retained_pinned:u64,
    allocations:u64,releases:u64,closed:bool,cancelled:bool,clean:bool,sticky_failure:bool,complete_owned_buffers:bool,unclassified_buffers:u64,
    driver_private:?u64=null,verbs_private:?u64=null,heap_and_checkpoint_residency:?u64=null,
    scope:[]const u8="actual_native_buffer_balance_not_whole_process_or_physical_leak_proof",
};
pub const Observer=struct{ctx:*anyopaque,acquire:*const fn(*anyopaque,u64,usize,Kind,u16) u64,commit:*const fn(*anyopaque,u64,u64,Kind)bool,claim:*const fn(*anyopaque,u64)bool,release:*const fn(*anyopaque,u64,bool) void,snapshot:?*const fn(*anyopaque)Ownership=null,epoch:u64=0};
pub const Mark=struct{observer:Observer,id:u64,
    fn current(m:Mark)bool{return if(active)|o|o.ctx==m.observer.ctx and o.epoch==m.observer.epoch else false;}
    pub fn claimed(m:Mark)bool{lock();defer mutex.unlock();return m.current() and m.observer.claim(m.observer.ctx,m.id);}
    pub fn committed(m:Mark,address:u64,kind:Kind)bool{lock();defer mutex.unlock();return m.current() and m.observer.commit(m.observer.ctx,m.id,address,kind);}
    pub fn finished(m:Mark,ok:bool)void{lock();defer mutex.unlock();if(m.current())m.observer.release(m.observer.ctx,m.id,ok);}
};
var active:?Observer=null;var mutex:std.atomic.Mutex=.unlocked;var epoch:u64=0;
threadlocal var owner:u16=std.math.maxInt(u16);
fn lock()void{while(!mutex.tryLock())std.Thread.yield() catch {};}
pub fn install(o:Observer)!void{lock();defer mutex.unlock();if(active!=null)return error.MemoryOwnerBusy;epoch=try std.math.add(u64,epoch,1);var selected=o;selected.epoch=epoch;active=selected;}
pub fn uninstall(ctx:*anyopaque)void{lock();defer mutex.unlock();std.debug.assert(active!=null and active.?.ctx==ctx);active=null;}
pub fn snapshot()?Ownership{lock();defer mutex.unlock();const o=active orelse return null;return if(o.snapshot)|get|get(o.ctx)else null;}
pub fn begin(n:usize,kind:Kind)!?Mark{
    if(n==0)return null;lock();defer mutex.unlock();const observed=active orelse return null;
    const id=observed.acquire(observed.ctx,0,n,kind,owner);if(id==0)return error.OwnershipRefused;
    return .{.observer=observed,.id=id};
}
pub const Scope=struct{previous:u16,pub fn end(s:Scope)void{owner=s.previous;}};
pub fn scope(tag:u16)Scope{const previous=owner;owner=tag;return .{.previous=previous};}
