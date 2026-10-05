//! Chapter 6: a random forest, built from our Chapter 5 tree.
//!
//! Many trees, each grown on a random sample of rows and choosing among
//! a random handful of features at every split. The forest's answer is
//! the average of the trees' probabilities.
use crate::AnyResult;
use crate::data::{SEED, TARGET, TEST, TRAIN, feature_names, load_csv, to_arrays};
use crate::metrics::{Confusion, auc, report};
use crate::tree::{Node, TreeSettings, add_gains, grow_tree, predict_one};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
pub const MODEL: &str = "forest_model.json";
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ForestSettings {
    pub n_trees: usize,
    pub max_features: usize,
    pub min_leaf: usize,
    pub max_depth: usize,
}
#[derive(Serialize, Deserialize)]
pub struct SavedForest {
    pub target: String,
    pub features: Vec<String>,
    pub settings: ForestSettings,
    pub threshold: f64,
    pub trees: Vec<Node>,
}
/// A grown forest, plus each training row's "out-of-bag" probability:
/// the average from only the trees that did NOT see that row.
pub struct Grown {
    pub trees: Vec<Node>,
    pub oob_probs: Vec<f64>,
    pub oob_actual: Vec<bool>,
}
// ---------- growing the forest ----------
pub fn grow_forest(x: &Array2<f64>, y: &Array1<bool>, settings: ForestSettings) -> Grown {
    let n = x.nrows();
    let tree_settings = TreeSettings {
        max_depth: settings.max_depth,
        min_leaf: settings.min_leaf,
        max_features: Some(settings.max_features),
    };
    // Trees are independent, so rayon builds them on all CPU cores at once.
    // Each tree gets its own seed, so the forest is identical on every run.
    let grown: Vec<(Node, Vec<bool>)> = (0..settings.n_trees)
        .into_par_iter()
        .map(|t| {
            let mut rng = StdRng::seed_from_u64(SEED + t as u64);
            // Bootstrap sample: n rows drawn WITH replacement.
            // Some rows appear twice or more; about a third never appear.
            let rows: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let mut in_bag = vec![false; n];
            for &r in &rows {
                in_bag[r] = true;
            }
            (grow_tree(x, y, &rows, tree_settings, &mut rng), in_bag)
        })
        .collect();
    // Out-of-bag: score each row using only the trees that never saw it.
    let mut sum = vec![0.0; n];
    let mut count = vec![0usize; n];
    for (tree, in_bag) in &grown {
        for i in 0..n {
            if !in_bag[i] {
                sum[i] += predict_one(tree, x.row(i));
                count[i] += 1;
            }
        }
    }
    let mut oob_probs = Vec::new();
    let mut oob_actual = Vec::new();
    for i in 0..n {
        if count[i] > 0 {
            oob_probs.push(sum[i] / count[i] as f64);
            oob_actual.push(y[i]);
        }
    }
    Grown {
        trees: grown.into_iter().map(|(tree, _)| tree).collect(),
        oob_probs,
        oob_actual,
    }
}
/// The forest's answer: the average of every tree's probability.
pub fn predict_probs(trees: &[Node], x: &Array2<f64>) -> Vec<f64> {
    x.rows()
        .into_iter()
        .map(|row| trees.iter().map(|t| predict_one(t, row)).sum::<f64>() / trees.len() as f64)
        .collect()
}
/// Share of trees whose own answer is at or above the threshold.
pub fn agreement(trees: &[Node], row: ndarray::ArrayView1<f64>, threshold: f64) -> f64 {
    let yes = trees
        .iter()
        .filter(|t| predict_one(t, row) >= threshold)
        .count();
    yes as f64 / trees.len() as f64
}
/// Each feature's share of all the impurity reduction in the forest.
fn importance(trees: &[Node], n_features: usize) -> Vec<f64> {
    let mut totals = vec![0.0; n_features];
    for tree in trees {
        add_gains(tree, &mut totals);
    }
    let sum: f64 = totals.iter().sum();
    totals.iter().map(|g| g / sum).collect()
}
fn best_threshold(probs: &[f64], actual: &[bool]) -> (f64, f64) {
    (1..=10)
        .map(|i| i as f64 * 5.0 / 100.0)
        .map(|t| (t, Confusion::new(t, probs, actual).f1()))
        .fold(
            (0.0, -1.0),
            |acc, cur| if cur.1 > acc.1 { cur } else { acc },
        )
}
fn save(
    features: Vec<String>,
    settings: ForestSettings,
    threshold: f64,
    trees: Vec<Node>,
) -> AnyResult<()> {
    let saved = SavedForest {
        target: TARGET.to_string(),
        features,
        settings,
        threshold,
        trees,
    };
    std::fs::write(MODEL, serde_json::to_string(&saved)?)?;
    println!("Saved: {MODEL}");
    Ok(())
}
// ---------- stages ----------
pub fn train() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let features = feature_names(&df);
    let (x, y) = to_arrays(&df, &features)?;
    // Common starting point: about the square root of the feature count per split.
    let settings = ForestSettings {
        n_trees: 300,
        max_features: 4,
        min_leaf: 10,
        max_depth: 64,
    };
    let grown = grow_forest(&x, &y, settings);
    println!(
        "Random forest: {} trees, {} features per split, min leaf {}",
        settings.n_trees, settings.max_features, settings.min_leaf
    );
    println!(
        "Out-of-bag rows scored: {} of {}",
        grown.oob_probs.len(),
        x.nrows()
    );
    println!(
        "Out-of-bag AUC: {:.3}",
        auc(&grown.oob_probs, &grown.oob_actual)
    );
    println!("\nFeature importance (share of all impurity reduction):");
    let mut ranked: Vec<(String, f64)> = features
        .iter()
        .cloned()
        .zip(importance(&grown.trees, features.len()))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, share) in &ranked {
        let bar = "#".repeat((share * 100.0).round() as usize);
        println!(
            "
{name:<16} {:>5.1}%
{bar}",
            share * 100.0
        );
    }
    println!();
    save(features, settings, 0.5, grown.trees)
}
pub fn evaluate() -> AnyResult<()> {
    let model: SavedForest = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let df = load_csv(TEST)?;
    let (x, y) = to_arrays(&df, &model.features)?;
    let probs = predict_probs(&model.trees, &x);
    let actual: Vec<bool> = y.to_vec();
    let baseline = actual.iter().filter(|&&a| !a).count() as f64 / actual.len() as f64;
    println!(
        "Forest: {} trees, {} features per split, min leaf {}",
        model.settings.n_trees, model.settings.max_features, model.settings.min_leaf
    );
    println!("Test rows: {}", actual.len());
    println!("Always-predict-0 accuracy: {:.2}%", baseline * 100.0);
    println!("AUC: {:.3}", auc(&probs, &actual));
    report(0.5, &probs, &actual);
    if model.threshold != 0.5 {
        report(model.threshold, &probs, &actual);
    }
    Ok(())
}
pub fn tune() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let features = feature_names(&df);
    let (x, y) = to_arrays(&df, &features)?;
    // Part 1: does adding trees help? (It never hurts; it levels off.)
    println!("More trees, same settings (4 features per split, min leaf 10):\n");
    println!("{:>6} {:>9}", "trees", "OOB AUC");
    for n_trees in [1, 5, 25, 100, 300, 600] {
        let settings = ForestSettings {
            n_trees,
            max_features: 4,
            min_leaf: 10,
            max_depth: 64,
        };
        let grown = grow_forest(&x, &y, settings);
        println!(
            "{n_trees:>6} {:>9.3}",
            auc(&grown.oob_probs, &grown.oob_actual)
        );
    }
    // Part 2: choose features-per-split and leaf size, judged out-of-bag.
    println!("\nGrid search, 300 trees each, judged on out-of-bag rows:\n");
    println!(
        "{:>9} {:>9} {:>7} {:>10} {:>7}",
        "features", "min leaf", "AUC", "best cut", "F1"
    );
    let mut best: Option<(ForestSettings, f64, f64)> = None;
    for max_features in [2, 3, 4, 6, 8, 14] {
        for min_leaf in [5, 10, 20, 40] {
            let settings = ForestSettings {
                n_trees: 300,
                max_features,
                min_leaf,
                max_depth: 64,
            };
            let grown = grow_forest(&x, &y, settings);
            let (cut, f1) = best_threshold(&grown.oob_probs, &grown.oob_actual);
            println!(
                "{max_features:>9} {min_leaf:>9} {:>7.3} {cut:>10.2} {f1:>7.3}",
                auc(&grown.oob_probs, &grown.oob_actual)
            );
            if best.as_ref().is_none_or(|b| f1 > b.2) {
                best = Some((settings, cut, f1));
            }
        }
    }
    let (settings, threshold, f1) = best.expect("grid is not empty");
    println!(
        "\nBest: {} features per split, min leaf {}, threshold {threshold:.2}, F1 {f1:.3}",
        settings.max_features, settings.min_leaf
    );
    let grown = grow_forest(&x, &y, settings);
    save(features, settings, threshold, grown.trees)?;
    println!("\nNow run: cargo run --release -- forest-evaluate");
    Ok(())
}
