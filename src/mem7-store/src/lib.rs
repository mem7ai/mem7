mod add;
mod builder;
pub mod constants;
mod crud;
pub mod decay;
mod engine;
pub(crate) mod graph_pipeline;
pub mod payload;
mod pipeline;
mod prompts;
pub(crate) mod rehearsal;
mod search;

use mem7_error::{Mem7Error, Result};

pub use builder::MemoryEngineBuilder;
pub use engine::MemoryEngine;

pub(crate) fn require_scope(
    operation: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
) -> Result<()> {
    if user_id.is_none() && agent_id.is_none() && run_id.is_none() {
        return Err(Mem7Error::Config(format!(
            "{operation} requires at least one of user_id, agent_id, or run_id"
        )));
    }
    Ok(())
}
