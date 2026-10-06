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
//!cargo run --release -- tree-train(Chapter 5: decision tree)
//!cargo run --release -- tree-evaluate(Chapter 5)
//!cargo run --release -- tree-tune(Chapter 5)
//!cargo run --release -- forest-train(Chapter 6: random forest)
//!cargo run --release -- forest-evaluate(Chapter 6)
//!cargo run --release -- forest-tune(Chapter 6)
//!cargo run --release -- text-split(Chapter 7: tweets and a linear SVM)
//!cargo run --release -- text-vocab(Chapter 7)
//!cargo run --release -- svm-train(Chapter 7)
//!cargo run --release -- svm-evaluate(Chapter 7)
//!cargo run --release -- svm-tune(Chapter 7)
//!cargo run --release -- svm-versions(Chapter 7)
//!cargo run --release -- svm-use(Chapter 7 (svm-use <n>))
//!cargo run --release -- svm-heart(Chapter 7)
//!cargo run --release -- nb-heart(Chapter 8: Naive Bayes and k-NN)
//!cargo run --release -- nb-tweets(Chapter 8)
//!cargo run --release -- knn-tune(Chapter 8)
//!cargo run --release -- knn-evaluate(Chapter 8)
//!cargo run --release -- nn-xor(Chapter 9: neural networks)
//!cargo run --release -- nn-heart(Chapter 9)
//!cargo run --release -- nn-digits(Chapter 9)
//!cargo run --release -- nn-show(Chapter 9 (nn-show <n>))
//!cargo run --release -- gpt-data(Chapter 10: a tiny GPT)
//!cargo run --release -- gpt-check(Chapter 10)
//!cargo run --release -- gpt-train(Chapter 10)
//!cargo run --release -- gpt-write(Chapter 10 (gpt-write "ROMEO:\n" 300 0.8))
//!cargo run --release -- gpt-attend(Chapter 10 (gpt-attend "text"))
//!cargo run --release -- llm-check(Chapter 11: an open model with Candle)
//!cargo run --release -- llm-write(Chapter 11 (llm-write "ROMEO:\n" 100 0.8 [tuned]))
//!cargo run --release -- llm-score(Chapter 11)
//!cargo run --release -- llm-train(Chapter 11)
mod bayes;
mod data;
mod explore;
mod forest;
mod gpt;
mod knn;
mod linear;
mod llm;
mod logistic;
mod metrics;
mod nn;
mod server;
mod svm;
mod text;
mod tree;
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
        "tree-train" => tree::train(),
        "tree-evaluate" => tree::evaluate(),
        "tree-tune" => tree::tune(),
        "forest-train" => forest::train(),
        "forest-evaluate" => forest::evaluate(),
        "forest-tune" => forest::tune(),
        "text-split" => text::split(),
        "text-vocab" => text::vocab(),
        "svm-train" => svm::train(),
        "svm-evaluate" => svm::evaluate(),
        "svm-tune" => svm::tune(),
        "svm-versions" => svm::versions(),
        "svm-use" => svm::use_version(),
        "svm-heart" => svm::heart(),
        "nb-heart" => bayes::heart(),
        "nb-tweets" => bayes::tweets(),
        "knn-tune" => knn::tune(),
        "knn-evaluate" => knn::evaluate(),
        "nn-xor" => nn::xor(),
        "nn-heart" => nn::heart(),
        "nn-digits" => nn::digits(),
        "nn-show" => nn::show(),
        "gpt-data" => gpt::data(),
        "gpt-check" => gpt::check(),
        "gpt-train" => gpt::train(),
        "gpt-write" => gpt::write(),
        "gpt-attend" => gpt::attend(),
        "llm-check" => llm::check(),
        "llm-write" => llm::write(),
        "llm-score" => llm::score(),
        "llm-train" => llm::train(),
        _ => {
            eprintln!("Usage: cargo run --release -- <explore|clean|split|linear-train|linear-
evaluate|train|evaluate|tune|serve|tree-train|tree-evaluate|tree-tune|forest-train|forest-evaluate|forest-
tune|text-split|text-vocab|svm-train|svm-evaluate|svm-tune|svm-versions|svm-use|svm-heart|nb-heart|nb-
tweets|knn-tune|knn-evaluate|nn-xor|nn-heart|nn-digits|nn-show|gpt-data|gpt-check|gpt-train|gpt-write|gpt-
attend|llm-check|llm-write|llm-score|llm-train>");
            std::process::exit(1);
        }
    }
}
