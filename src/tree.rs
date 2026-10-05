//! Chapter 5: a decision tree, written from scratch.
//!
//! Each leaf stores the SHARE of class-1 rows that ended up there,
//! so the tree gives a probability, just like logistic regression.
use crate::AnyResult;
use crate::data::{
    FOLDS, SEED, TARGET, TEST, TRAIN, assign_folds, feature_names, load_csv, to_arrays,
};
use crate::metrics::{Confusion, auc, report};
use ndarray::{Array1, Array2, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
pub const MODEL: &str = "tree_model.json";
/// One point in the tree: either a question, or an answer.
#[derive(Serialize, Deserialize)]
pub enum Node {
    /// "Is feature <= threshold?"Yes -> left, No -> right.
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
        /// How much this question reduced impurity, weighted by rows (for feature importance).
        #[serde(default)]
        gain: f64,
    },
    /// An answer: the share of class-1 training rows that reached here.
    Leaf { probability: f64, rows: usize },
}
/// Limits that stop the tree from growing until it memorizes the data.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TreeSettings {
    pub max_depth: usize,
    pub min_leaf: usize,
    /// How many features to consider at each split. None = all (a single tree).
    /// A random forest sets this lower so its trees differ from each other.
    #[serde(default)]
    pub max_features: Option<usize>,
}
#[derive(Serialize, Deserialize)]
pub struct SavedTree {
    pub target: String,
    pub features: Vec<String>,
    pub settings: TreeSettings,
    pub threshold: f64,
    pub root: Node,
}
// ---------- building the tree ----------
/// Gini impurity of a group: 0 = all one class, 0.5 = a 50/50 mix.
fn gini(ones: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let p = ones as f64 / total as f64;
    2.0 * p * (1.0 - p)
}
struct BestSplit {
    feature: usize,
    threshold: f64,
    impurity: f64,
}
/// Try every candidate feature and every cut point; keep the one that leaves
/// the two sides purest (lowest weighted Gini).
fn find_best_split(
    x: &Array2<f64>,
    y: &Array1<bool>,
    rows: &[usize],
    candidates: &[usize],
    min_leaf: usize,
) -> Option<BestSplit> {
    let n = rows.len();
    let total_ones = rows.iter().filter(|&&r| y[r]).count();
    let mut best: Option<BestSplit> = None;
    for &feature in candidates {
        // Sort this group's rows by the feature's value.
        let mut sorted: Vec<usize> = rows.to_vec();
        sorted.sort_by(|&a, &b| x[[a, feature]].total_cmp(&x[[b, feature]]));
        // Walk through, moving one row at a time from right side to left side.
        let mut left_ones = 0;
        for i in 0..n - 1 {
            if y[sorted[i]] {
                left_ones += 1;
            }
            let left_n = i + 1;
            let right_n = n - left_n;
            let here = x[[sorted[i], feature]];
            let next = x[[sorted[i + 1], feature]];
            // Only cut between two different values, and keep both sides big enough.
            if here == next || left_n < min_leaf || right_n < min_leaf {
                continue;
            }
            let impurity = (left_n as f64 * gini(left_ones, left_n)
                + right_n as f64 * gini(total_ones - left_ones, right_n))
                / n as f64;
            if best.as_ref().is_none_or(|b| impurity < b.impurity) {
                best = Some(BestSplit {
                    feature,
                    threshold: (here + next) / 2.0,
                    impurity,
                });
            }
        }
    }
    best
}
fn build(
    x: &Array2<f64>,
    y: &Array1<bool>,
    rows: &[usize],
    depth: usize,
    settings: TreeSettings,
    rng: &mut StdRng,
) -> Node {
    let ones = rows.iter().filter(|&&r| y[r]).count();
    let leaf = Node::Leaf {
        probability: ones as f64 / rows.len() as f64,
        rows: rows.len(),
    };
    // Stop: deep enough, too few rows, or already pure.
    if depth >= settings.max_depth
        || rows.len() < 2 * settings.min_leaf
        || ones == 0
        || ones == rows.len()
    {
        return leaf;
    }
    // Which features may this split use? All of them, or a random handful.
    let candidates: Vec<usize> = match settings.max_features {
        Some(k) if k < x.ncols() => rand::seq::index::sample(rng, x.ncols(), k).into_vec(),
        _ => (0..x.ncols()).collect(),
    };
    let Some(best) = find_best_split(x, y, rows, &candidates, settings.min_leaf) else {
        return leaf;
    };
    // Stop if the split does not make the groups any purer.
    let parent = gini(ones, rows.len());
    if best.impurity >= parent {
        return leaf;
    }
    let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
        .iter()
        .partition(|&&r| x[[r, best.feature]] <= best.threshold);
    Node::Split {
        feature: best.feature,
        threshold: best.threshold,
        left: Box::new(build(x, y, &left_rows, depth + 1, settings, rng)),
        right: Box::new(build(x, y, &right_rows, depth + 1, settings, rng)),
        gain: rows.len() as f64 * (parent - best.impurity),
    }
}
/// Grow a tree on any list of rows (the forest passes a random sample).
pub fn grow_tree(
    x: &Array2<f64>,
    y: &Array1<bool>,
    rows: &[usize],
    settings: TreeSettings,
    rng: &mut StdRng,
) -> Node {
    build(x, y, rows, 0, settings, rng)
}
/// A single tree: every row, every feature.
pub fn fit_tree(x: &Array2<f64>, y: &Array1<bool>, settings: TreeSettings) -> Node {
    let rows: Vec<usize> = (0..x.nrows()).collect();
    let mut rng = StdRng::seed_from_u64(SEED);
    grow_tree(x, y, &rows, settings, &mut rng)
}
/// Add up each feature's gain across the whole tree.
pub fn add_gains(node: &Node, totals: &mut [f64]) {
    if let Node::Split {
        feature,
        left,
        right,
        gain,
        ..
    } = node
    {
        totals[*feature] += gain;
        add_gains(left, totals);
        add_gains(right, totals);
    }
}
// ---------- using the tree ----------
/// Walk from the root down to a leaf, answering each question.
pub fn predict_one(node: &Node, row: ArrayView1<f64>) -> f64 {
    match node {
        Node::Leaf { probability, .. } => *probability,
        Node::Split {
            feature,
            threshold,
            left,
            right,
            ..
        } => {
            if row[*feature] <= *threshold {
                predict_one(left, row)
            } else {
                predict_one(right, row)
            }
        }
    }
}
pub fn predict_probs(root: &Node, x: &Array2<f64>) -> Vec<f64> {
    x.rows()
        .into_iter()
        .map(|row| predict_one(root, row))
        .collect()
}
/// Walk down the tree like predict_one, but write down each answer on the way.
pub fn explain(root: &Node, row: &[f64], features: &[String]) -> (f64, Vec<String>) {
    let mut node = root;
    let mut path = Vec::new();
    loop {
        match node {
            Node::Leaf { probability, .. } => return (*probability, path),
            Node::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                let name = &features[*feature];
                let value = row[*feature];
                if value <= *threshold {
                    path.push(format!("{name} {value} <= {threshold:.1}"));
                    node = left;
                } else {
                    path.push(format!("{name} {value} > {threshold:.1}"));
                    node = right;
                }
            }
        }
    }
}
/// Print the tree as indented if/else rules.
fn print_node(node: &Node, features: &[String], indent: usize) {
    let pad = "
"
    .repeat(indent);
    match node {
        Node::Leaf { probability, rows } => {
            println!(
                "{pad}-> risk {:.1}%
({rows} people)",
                probability * 100.0
            );
        }
        Node::Split {
            feature,
            threshold,
            left,
            right,
            ..
        } => {
            println!("{pad}if {} <= {threshold:.1}:", features[*feature]);
            print_node(left, features, indent + 1);
            println!("{pad}else:");
            print_node(right, features, indent + 1);
        }
    }
}
fn count_leaves(node: &Node) -> usize {
    match node {
        Node::Leaf { .. } => 1,
        Node::Split { left, right, .. } => count_leaves(left) + count_leaves(right),
    }
}
// ---------- stages ----------
pub fn train() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let features = feature_names(&df);
    let (x, y) = to_arrays(&df, &features)?;
    let settings = TreeSettings {
        max_depth: 3,
        min_leaf: 20,
        max_features: None,
    };
    let root = fit_tree(&x, &y, settings);
    println!(
        "Decision tree (max depth {}, min {} people per leaf):\n",
        settings.max_depth, settings.min_leaf
    );
    print_node(&root, &features, 0);
    println!("\nLeaves: {}", count_leaves(&root));
    let saved = SavedTree {
        target: TARGET.to_string(),
        features,
        settings,
        threshold: 0.5,
        root,
    };
    std::fs::write(MODEL, serde_json::to_string_pretty(&saved)?)?;
    println!("Saved: {MODEL}");
    Ok(())
}
pub fn evaluate() -> AnyResult<()> {
    let model: SavedTree = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let df = load_csv(TEST)?;
    let (x, y) = to_arrays(&df, &model.features)?;
    let probs = predict_probs(&model.root, &x);
    let actual: Vec<bool> = y.to_vec();
    let baseline = actual.iter().filter(|&&a| !a).count() as f64 / actual.len() as f64;
    println!(
        "Tree: max depth {}, min leaf {}, leaves {}",
        model.settings.max_depth,
        model.settings.min_leaf,
        count_leaves(&model.root)
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
    let actual: Vec<bool> = y.to_vec();
    let fold_of = assign_folds(&y);
    let depths = [2, 3, 4, 5, 6, 8, 10];
    let min_leaves = [5, 20, 50];
    let thresholds: Vec<f64> = (1..=10).map(|i| i as f64 * 5.0 / 100.0).collect();
    let mut best = (
        TreeSettings {
            max_depth: 0,
            min_leaf: 0,
            max_features: None,
        },
        0.0,
        -1.0,
    );
    println!("5-fold cross-validation on the training data only\n");
    println!(
        "{:>6} {:>9} {:>7} {:>10} {:>7}",
        "depth", "min leaf", "AUC", "best cut", "F1"
    );
    for &max_depth in &depths {
        for &min_leaf in &min_leaves {
            let settings = TreeSettings {
                max_depth,
                min_leaf,
                max_features: None,
            };
            let mut probs = vec![0.0; actual.len()];
            for fold in 0..FOLDS {
                let fit_rows: Vec<usize> =
                    (0..actual.len()).filter(|&i| fold_of[i] != fold).collect();
                let check_rows: Vec<usize> =
                    (0..actual.len()).filter(|&i| fold_of[i] == fold).collect();
                let x_fit = x.select(ndarray::Axis(0), &fit_rows);
                let y_fit = y.select(ndarray::Axis(0), &fit_rows);
                let x_check = x.select(ndarray::Axis(0), &check_rows);
                let root = fit_tree(&x_fit, &y_fit, settings);
                for (&row, p) in check_rows.iter().zip(predict_probs(&root, &x_check)) {
                    probs[row] = p;
                }
            }
            let (cut, f1) = thresholds
                .iter()
                .map(|&t| (t, Confusion::new(t, &probs, &actual).f1()))
                .fold(
                    (0.0, -1.0),
                    |acc, cur| if cur.1 > acc.1 { cur } else { acc },
                );
            println!(
                "{max_depth:>6} {min_leaf:>9} {:>7.3} {cut:>10.2} {f1:>7.3}",
                auc(&probs, &actual)
            );
            if f1 > best.2 {
                best = (settings, cut, f1);
            }
        }
    }
    let (settings, threshold, f1) = best;
    println!(
        "\nBest: depth {}, min leaf {}, threshold {threshold:.2}, F1 {f1:.3}",
        settings.max_depth, settings.min_leaf
    );
    let saved = SavedTree {
        target: TARGET.to_string(),
        features: features.clone(),
        settings,
        threshold,
        root: fit_tree(&x, &y, settings),
    };
    std::fs::write(MODEL, serde_json::to_string_pretty(&saved)?)?;
    println!("Saved: {MODEL}");
    println!("\nNow run: cargo run -- tree-evaluate");
    Ok(())
}
