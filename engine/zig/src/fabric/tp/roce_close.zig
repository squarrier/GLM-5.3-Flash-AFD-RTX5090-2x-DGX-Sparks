//! Typed bounded-close disposition; failed or unproven native registrations remain quarantined.
pub const Reason=enum { closed,failed,deadline,proxy,gpu,wire,missing_completion };
pub const Receipt=struct {quiet: bool,retained_bytes: usize,reason: Reason};
