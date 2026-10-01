//! Machine learning with Rust: one program, one stage at a time.
//!
//! Run a stage by name:
//!cargo run --release -- explore (Chapter 1: a first look at the data)

mod explore;

/// Any error from any library can be returned with `?`.
pub type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;


fn main() -> AnyResult<()> {
    let stage = std::env::args().nth(1).unwrap_or_default();
    match stage.as_str() {
        "explore" => explore::explore(),
        _ => {
            eprint!("Usage: cargo run --release -- <explore>");
            std::process::exit(1);
        }
        
    }
}
