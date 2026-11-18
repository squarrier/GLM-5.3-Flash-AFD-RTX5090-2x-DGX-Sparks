// SPDX-License-Identifier: Apache-2.0
//! Bounded native-buffer event ledger. Original: Helios/OpenAI Codex for Scott Quarrier.
//! Counts actual owning allocations and successful releases; quiet is not a release.
const std=@import("std");const cuda=@import("cuda");const m=@import("cuda_hybrid_memory.zig");const witness=cuda.memory_owner;
const State=enum{empty,reserved,live,releasing,retained,released};
pub fn scopeOwner(owner:m.Owner)witness.Scope{return witness.scope(@intFromEnum(owner));}
const Item=struct{id:u64=0,address:u64=0,n:u64=0,kind:witness.Kind=.device,owner:?m.Owner=null,state:State=.empty};
pub const Status=struct{
    owners:[@typeInfo(m.Owner).@"enum".field_names.len]m.Charge=@splat(.{}),
    schema:u32=1,role:m.Role,incarnation:u128,live:m.Charge=. {},retained:m.Charge=. {},pending:m.Charge=. {},
    allocations:u64,releases:u64,sticky_failure:bool,closed:bool,cancelled:bool,clean:bool,
    complete_owned_buffers:bool,unclassified_buffers:u64=0,hooked_logical_bytes:u64=0,
    driver_private:?u64=null,verbs_private:?u64=null,heap_and_checkpoint_residency:?u64=null,
    scope:[]const u8="actual_native_buffer_balance_not_whole_process_or_physical_leak_proof",
};
pub const Ledger=struct{
    role:m.Role,incarnation:u128,items:[m.ledger_capacity]Item=@splat(.{}),sequence:u64=0,allocations:u64=0,releases:u64=0,
    closed:bool=false,cancelled:bool=false,sticky_failure:bool=false,overflow:bool=false,
    refusal_ctx:?*anyopaque=null,refuse:?*const fn(*anyopaque)void=null,
    mutex:std.atomic.Mutex=.unlocked,
    pub fn init(role:m.Role,incarnation:u128)Ledger{return .{.role=role,.incarnation=incarnation};}
    fn lock(l:*Ledger)void{while(!l.mutex.tryLock())std.Thread.yield() catch {};}
    pub fn install(l:*Ledger)!void{if(l.incarnation==0)return error.IdentityMismatch;try witness.install(.{.ctx=l,.acquire=acquire,.commit=commit,.claim=claim,.release=release,.snapshot=reported});}
    fn reported(ctx:*anyopaque)witness.Ownership{
        const l:*Ledger=@ptrCast(@alignCast(ctx));const s=l.status();return .{.role=@tagName(s.role),.incarnation=s.incarnation,.live_device=s.live.device,.live_pinned=s.live.pinned,.retained_device=s.retained.device,.retained_pinned=s.retained.pinned,.allocations=s.allocations,.releases=s.releases,.closed=s.closed,.cancelled=s.cancelled,.clean=s.clean,.sticky_failure=s.sticky_failure,.complete_owned_buffers=s.complete_owned_buffers,.unclassified_buffers=s.unclassified_buffers};
    }
    pub fn uninstall(l:*Ledger)void{witness.uninstall(l);}
    pub fn scope(_: *Ledger,owner:m.Owner)witness.Scope{return scopeOwner(owner);}
    fn acquire(ctx:*anyopaque,address:u64,n:usize,kind:witness.Kind,owner:u16)u64{
        const l:*Ledger=@ptrCast(@alignCast(ctx));l.lock();defer l.mutex.unlock();
        if(l.closed or l.sticky_failure or l.overflow){l.sticky_failure=true;l.overflow=true;if(l.refuse)|f|f(l.refusal_ctx.?);return 0;}
        const id=std.math.add(u64,l.sequence,1) catch {l.sticky_failure=true;l.overflow=true;if(l.refuse)|f|f(l.refusal_ctx.?);return 0;};
        for(&l.items)|*item|if(item.state==.empty or item.state==.released){
            item.*=.{.id=id,.address=address,.n=n,.kind=kind,.owner=std.enums.fromInt(m.Owner,owner),.state=.reserved};l.sequence=id;return id;
        };
        l.overflow=true;l.sticky_failure=true;if(l.refuse)|f|f(l.refusal_ctx.?);return 0;
    }
    fn commit(ctx:*anyopaque,id:u64,address:u64,kind:witness.Kind)bool{
        const l:*Ledger=@ptrCast(@alignCast(ctx));l.lock();defer l.mutex.unlock();
        for(&l.items)|*item|if(item.id==id and item.state==.reserved){item.address=address;item.kind=kind;item.state=.live;l.allocations+=1;if(l.closed or l.sticky_failure){l.sticky_failure=true;if(l.refuse)|f|f(l.refusal_ctx.?);return false;}return true;};
        l.sticky_failure=true;l.overflow=true;return false;
    }
    fn claim(ctx:*anyopaque,id:u64)bool{
        const l:*Ledger=@ptrCast(@alignCast(ctx));l.lock();defer l.mutex.unlock();
        for(&l.items)|*item|if(item.id==id and id!=0 and (item.state==.live or item.state==.retained)){item.state=.releasing;return true;};
        l.sticky_failure=true;if(l.refuse)|f|f(l.refusal_ctx.?);return false;
    }
    fn release(ctx:*anyopaque,id:u64,ok:bool)void{
        const l:*Ledger=@ptrCast(@alignCast(ctx));l.lock();defer l.mutex.unlock();
        for(&l.items)|*item|if(item.id==id and id!=0){
            if(item.state==.reserved and ok){item.state=.released;return;}
            if(item.state!=.releasing){l.sticky_failure=true;return;}
            if(ok){item.state=.released;l.releases+=1;}else{item.state=.retained;l.sticky_failure=true;if(l.refuse)|f|f(l.refusal_ctx.?);}return;
        };
        l.sticky_failure=true;l.overflow=true;
    }
    pub fn finish(l:*Ledger,quiet:bool,cancelled:bool)void{
        l.lock();defer l.mutex.unlock();l.closed=true;l.cancelled=l.cancelled or cancelled;
        if(!quiet){l.sticky_failure=true;for(&l.items)|*i|if(i.state==.live){i.state=.retained;};}
    }
    pub fn status(l:*Ledger)Status{
        l.lock();defer l.mutex.unlock();var out=Status{.role=l.role,.incarnation=l.incarnation,.allocations=l.allocations,.releases=l.releases,.sticky_failure=l.sticky_failure,.closed=l.closed,.cancelled=l.cancelled,.clean=l.closed and !l.overflow,.complete_owned_buffers=!l.overflow};
        for(l.items)|item|{
            if(item.state!=.live and item.state!=.retained and item.state!=.reserved and item.state!=.releasing)continue;
            out.clean=false;if(item.owner==null){out.unclassified_buffers+=1;out.complete_owned_buffers=false;}
            if(item.kind==.hooked_range){out.hooked_logical_bytes+=item.n;out.complete_owned_buffers=false;continue;}
            const c:m.Charge=if(item.kind==.device).{.device=item.n} else .{.host=item.n,.pinned=item.n};
            if(item.owner)|owner|{const at=@intFromEnum(owner);out.owners[at]=out.owners[at].plus(c) catch {out.complete_owned_buffers=false;out.clean=false;return out;};}
            const target=if(item.state==.reserved)&out.pending else if(item.state==.live or item.state==.releasing)&out.live else &out.retained;
            target.*=target.plus(c) catch {out.complete_owned_buffers=false;out.clean=false;return out;};
        }return out;
    }
};
