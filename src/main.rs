//! Machine learning with Rust: one program, one stage at a time.
//!
//! Run a stage by name:
//!cargo run --release -- explore(Chapter 1: a first look at the data)
//!cargo run --release -- clean(Chapter 2: data wrangling)
//!cargo run --release -- split(Chapter 2)
//!cargo run --release -- linear-train(Chapter 3: linear regression)
//!cargo run --release -- linear-evaluate(Chapter 3)
//!cargo run --release -- train(Chapter 4: logistic regression)
//!cargo run --release -- evaluate(Chapter 4)
//!cargo run --release -- tune(Chapter 4)
//!cargo run --release -- serve(Chapter 4 onward: serve every model)

mod data;
mod explore;
mod linear;
mod logistic;
mod metrics;
mod server;

/// Any error from any library can be returned with `?`.
pub type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> AnyResult<()> {
    let stage = std::env::args().nth(1).unwrap_or_default();
    match stage.as_str() {
        "explore" => explore::explore(),
        "clean" => data::clean(),
        "split" => data::split(),
        "linear-train" => linear::train(),
        "linear-evaluate" => linear::evaluate(),
        "train" => logistic::train(),
        "evaluate" => logistic::evaluate(),
        "tune" => logistic::tune(),
        "serve" => server::serve(),
        _ => {
            eprint!(
                "Usage: cargo run --release -- <explore|clean|split|linear-train|linear-
evaluate|train|evaluate|tune|serve>"
            );
            std::process::exit(1);
        }
    }
}
