//! grammar_server: what a tf-glm-afd built without -Dgram2abi imports as "grammar_abi": no loader and no ownership, so the
//! attention role's TF_AFD_GRAMMAR_SERVE=1 refuses every structured request by name (afd_attn_grammar.zig).
pub const absent = "this tf-glm-afd was built without grammar_abi's loader (-Dgram2abi)";
