//! Chapter 7: a linear support vector machine, trained from scratch.
//!
//! One yes/no SVM per sentiment ("Positive or not?", "Negative or not?",
//! "Neutral or not?"). The class whose SVM gives the highest score wins.
//! Each SVM is trained with Pegasos: many small corrections, one tweet at
//! a time, the same way neural networks learn.
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AnyResult;
use crate::data::{self, FOLDS, SEED};
use crate::metrics::auc;
use crate::text::{self, CLASSES, Sparse, Tweet, Vocabulary};

pub const MODELS_DIR: &str = "models/svm";
pub const CATALOG: &str = "models/svm/catalog.json";

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct SvmSettings {
    /// Regularization: higher = simpler model, wider margin, more mistakes allowed.
    pub lambda: f64,
    /// Passes over the training tweets.
    pub epochs: usize,
}

// One yes/no SVM: score = weights · x + bias. Positive score = "yes".
#[derive(Serialize, Deserialize, Clone)]
pub struct BinarySvm {
    pub weights: Vec<f64>,
    pub bias: f64,
}
impl BinarySvm {
    pub fn score(&self, x: &Sparse) -> f64 {
        x.iter().map(|&(i, v)| self.weights[i] * v).sum::<f64>() + self.bias
    }
}
/// Everything needed to classify new text, plus where it came from.
#[derive(Serialize, Deserialize)]
pub struct SavedSvm {
    pub version: usize,
    pub trained_unix: u64,
    pub settings: SvmSettings,
    pub train_rows: usize,
    pub note: String,
    pub classes: Vec<String>,
    pub vocab: Vocabulary,
    pub models: Vec<BinarySvm>,
}
impl SavedSvm {
    pub fn scores(&self, x: &Sparse) -> Vec<f64> {
        self.models.iter().map(|m| m.score(x)).collect()
    }
    pub fn predict(&self, text: &str) -> (usize, Vec<f64>) {
        let scores = self.scores(&self.vocab.transform(text));
        (argmax(&scores), scores)
    }
    /// The words in `text` that pushed hardest towards `class`.
    pub fn top_words(&self, text: &str, class: usize, n: usize) -> Vec<String> {
        let x = self.vocab.transform(text);
        let weights = &self.models[class].weights;
        let mut pushes: Vec<(usize, f64)> = x
            .iter()
            .map(|&(i, v)| (i, weights[i] * v))
            .filter(|(_, p)| *p > 0.0)
            .collect();
        pushes.sort_by(|a, b| b.1.total_cmp(&a.1));
        pushes
            .iter()
            .take(n)
            .map(|(i, _)| self.vocab.words[*i].clone())
            .collect()
    }
}
pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
// ---------- training (Pegasos) ----------
/// Training state for one yes/no SVM.
/// The weights are stored as `scale × v`, so shrinking every weight
/// (the regularization step) is one multiplication instead of thousands.
///
/// During the last pass it also keeps a running average of the weights.
/// Single steps jump around; their average is far steadier, and that
/// average is the model we keep.
struct Learner {
    v: Vec<f64>,
    bias_v: f64,
    scale: f64,
    averaging: bool,
    steps_averaged: usize,
    sum_scale: f64, // total of `scale` over the averaged steps
    r: Vec<f64>,    // bookkeeping so the average costs nothing extra per step
    r_bias: f64,
    bank: Vec<f64>,
    // sums set aside whenever `scale` is reset
    bank_bias: f64,
}
impl Learner {
    fn new(dims: usize) -> Self {
        Learner {
            v: vec![0.0; dims],
            bias_v: 0.0,
            scale: 1.0,
            averaging: false,
            steps_averaged: 0,
            sum_scale: 0.0,
            r: vec![0.0; dims],
            r_bias: 0.0,
            bank: vec![0.0; dims],
            bank_bias: 0.0,
        }
    }
    fn score(&self, x: &Sparse) -> f64 {
        self.scale * (x.iter().map(|&(i, v)| self.v[i] * v).sum::<f64>() + self.bias_v)
    }
    /// One Pegasos step on one example. `y` is +1 (this class) or -1 (not).
    fn step(&mut self, x: &Sparse, y: f64, lambda: f64, t: usize) {
        let eta = 1.0 / (lambda * (t as f64 + 1.0)); // learning rate shrinks over time
        let margin = y * self.score(x);
        // Shrink all weights a little (keeps the model simple).
        self.scale *= 1.0 - eta * lambda;
        // Inside the margin or on the wrong side? Nudge towards the right answer.
        if margin < 1.0 {
            for &(i, value) in x {
                let change = eta * y * value / self.scale;
                self.v[i] += change;
                self.r[i] += change * self.sum_scale;
            }
            let change = eta * y / self.scale;
            self.bias_v += change;
            self.r_bias += change * self.sum_scale;
        }
        if self.averaging {
            self.sum_scale += self.scale;
            self.steps_averaged += 1;
        }
        // Keep the numbers in a safe range.
        if self.scale < 1e-9 {
            self.bank_sums();
            for w in self.v.iter_mut() {
                *w *= self.scale;
            }
            self.bias_v *= self.scale;
            self.scale = 1.0;
        }
    }
    /// Move the running sums into `bank` before `v` changes units.
    fn bank_sums(&mut self) {
        for i in 0..self.v.len() {
            self.bank[i] += self.sum_scale * self.v[i] - self.r[i];
            self.r[i] = 0.0;
        }
        self.bank_bias += self.sum_scale * self.bias_v - self.r_bias;
        self.r_bias = 0.0;
        self.sum_scale = 0.0;
    }
    fn start_averaging(&mut self) {
        self.averaging = true;
        self.r.iter_mut().for_each(|r| *r = 0.0);
        self.r_bias = 0.0;
    }
    fn finish(mut self) -> BinarySvm {
        if self.steps_averaged == 0 {
            return BinarySvm {
                weights: self.v.iter().map(|w| w * self.scale).collect(),
                bias: self.bias_v * self.scale,
            };
        }
        self.bank_sums();
        let n = self.steps_averaged as f64;
        BinarySvm {
            weights: self.bank.iter().map(|w| w / n).collect(),
            bias: self.bank_bias / n,
        }
    }
}
/// Train one SVM per class, side by side. With `verbose`, print progress after each pass.
pub fn fit(
    xs: &[Sparse],
    labels: &[usize],
    dims: usize,
    settings: SvmSettings,
    verbose: bool,
) -> Vec<BinarySvm> {
    let n_classes = labels.iter().max().map_or(0, |m| m + 1);
    let mut learners: Vec<Learner> = (0..n_classes).map(|_| Learner::new(dims)).collect();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let mut t = 0;
    for epoch in 1..=settings.epochs {
        order.shuffle(&mut rng);
        if epoch == settings.epochs {
            learners.iter_mut().for_each(Learner::start_averaging);
        }
        for &i in &order {
            t += 1;
            for (class, learner) in learners.iter_mut().enumerate() {
                let y = if labels[i] == class { 1.0 } else { -1.0 };
                learner.step(&xs[i], y, settings.lambda, t);
            }
        }
        if verbose {
            let correct = (0..xs.len())
                .filter(|&i| {
                    let scores: Vec<f64> = learners.iter().map(|l| l.score(&xs[i])).collect();
                    argmax(&scores) == labels[i]
                })
                .count();
            println!(
                "
pass {epoch:>2}: training accuracy {:.2}%",
                correct as f64 / xs.len() as f64 * 100.0
            );
        }
    }
    learners.into_iter().map(Learner::finish).collect()
}
// ---------- scoring ----------
pub struct Report {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub confusion: [[usize; 3]; 3], // [actual][predicted]
}
pub fn score_predictions(actual: &[usize], predicted: &[usize]) -> Report {
    let mut confusion = [[0usize; 3]; 3];
    for (&a, &p) in actual.iter().zip(predicted) {
        confusion[a][p] += 1;
    }
    let correct: usize = (0..3).map(|c| confusion[c][c]).sum();
    let f1s: Vec<f64> = (0..3).map(|c| class_scores(&confusion, c).2).collect();
    Report {
        accuracy: correct as f64 / actual.len() as f64,
        macro_f1: f1s.iter().sum::<f64>() / 3.0,
        confusion,
    }
}
/// (precision, recall, F1) for one class, treating it as "yes" and the rest as "no".
fn class_scores(confusion: &[[usize; 3]; 3], class: usize) -> (f64, f64, f64) {
    let tp = confusion[class][class] as f64;
    let predicted: f64 = (0..3).map(|a| confusion[a][class] as f64).sum();
    let actual: f64 = confusion[class].iter().sum::<usize>() as f64;
    let precision = if predicted > 0.0 { tp / predicted } else { 0.0 };
    let recall = if actual > 0.0 { tp / actual } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}
// ---------- versions ----------
#[derive(Serialize, Deserialize)]
pub struct CatalogEntry {
    pub version: usize,
    pub file: String,
    pub trained_unix: u64,
    pub lambda: f64,
    pub epochs: usize,
    pub note: String,
}
#[derive(Serialize, Deserialize, Default)]
pub struct Catalog {
    pub active: usize,
    pub versions: Vec<CatalogEntry>,
}
pub fn load_catalog() -> AnyResult<Catalog> {
    match std::fs::read_to_string(CATALOG) {
        Ok(json) => Ok(serde_json::from_str(&json)?),
        Err(_) => Ok(Catalog::default()),
    }
}
fn save_catalog(catalog: &Catalog) -> AnyResult<()> {
    std::fs::write(CATALOG, serde_json::to_string_pretty(catalog)?)?;
    Ok(())
}
/// Save a new model as the next version and make it the active one.
fn save_new_version(
    vocab: Vocabulary,
    models: Vec<BinarySvm>,
    settings: SvmSettings,
    train_rows: usize,
    note: &str,
) -> AnyResult<usize> {
    std::fs::create_dir_all(MODELS_DIR)?;
    let mut catalog = load_catalog()?;
    let version = catalog
        .versions
        .iter()
        .map(|v| v.version)
        .max()
        .unwrap_or(0)
        + 1;
    let file = format!("{MODELS_DIR}/v{version}.json");
    let trained_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let saved = SavedSvm {
        version,
        trained_unix,
        settings,
        train_rows,
        note: note.to_string(),
        classes: CLASSES.iter().map(|c| c.to_string()).collect(),
        vocab,
        models,
    };
    std::fs::write(&file, serde_json::to_string(&saved)?)?;
    catalog.versions.push(CatalogEntry {
        version,
        file: file.clone(),
        trained_unix,
        lambda: settings.lambda,
        epochs: settings.epochs,
        note: note.to_string(),
    });
    catalog.active = version;
    save_catalog(&catalog)?;
    println!("Saved: {file} (now active)");
    Ok(version)
}
pub fn load_active() -> AnyResult<SavedSvm> {
    let catalog = load_catalog()?;
    let entry = catalog
        .versions
        .iter()
        .find(|v| v.version == catalog.active)
        .ok_or("no active SVM version; run svm-train first")?;
    let mut model: SavedSvm = serde_json::from_str(&std::fs::read_to_string(&entry.file)?)?;
    model.vocab.build_index();
    Ok(model)
}
// ---------- stages ----------
fn vectorize(vocab: &Vocabulary, tweets: &[Tweet]) -> Vec<Sparse> {
    tweets.iter().map(|t| vocab.transform(&t.text)).collect()
}
pub fn train() -> AnyResult<()> {
    let tweets = text::load(text::TRAIN)?;
    let texts: Vec<&str> = tweets.iter().map(|t| t.text.as_str()).collect();
    let vocab = Vocabulary::fit(&texts);
    let xs = vectorize(&vocab, &tweets);
    let labels: Vec<usize> = tweets.iter().map(|t| t.label).collect();
    let settings = SvmSettings {
        lambda: 1e-4,
        epochs: 10,
    };
    println!(
        "Training 3 linear SVMs on {} tweets, {} words, lambda {}",
        tweets.len(),
        vocab.len(),
        settings.lambda
    );
    let models = fit(&xs, &labels, vocab.len(), settings, true);
    save_new_version(vocab, models, settings, tweets.len(), "default settings")?;
    Ok(())
}
pub fn evaluate() -> AnyResult<()> {
    let model = load_active()?;
    let tweets = text::load(text::TEST)?;
    let actual: Vec<usize> = tweets.iter().map(|t| t.label).collect();
    let predicted: Vec<usize> = tweets.iter().map(|t| model.predict(&t.text).0).collect();
    let report = score_predictions(&actual, &predicted);
    let mut counts = [0usize; 3];
    for &a in &actual {
        counts[a] += 1;
    }
    let most_common = argmax(&counts.map(|c| c as f64));
    println!(
        "SVM version {} (lambda {}, {} passes)",
        model.version, model.settings.lambda, model.settings.epochs
    );
    println!("Test tweets: {}", tweets.len());
    println!(
        "Always-predict-{} accuracy: {:.2}%",
        CLASSES[most_common],
        counts[most_common] as f64 / tweets.len() as f64 * 100.0
    );
    println!("Accuracy: {:.2}%", report.accuracy * 100.0);
    println!("Macro F1: {:.3}", report.macro_f1);
    println!(
        "\n{:<10} {:>9} {:>7} {:>6}",
        "Class", "Precision", "Recall", "F1"
    );
    for (c, name) in CLASSES.iter().enumerate() {
        let (p, r, f1) = class_scores(&report.confusion, c);
        println!(
            "{name:<10} {:>8.1}% {:>6.1}% {f1:>6.3}",
            p * 100.0,
            r * 100.0
        );
    }
    println!(
        "\n{:<18}{:>9}{:>9}{:>9}",
        "", "Negative", "Neutral", "Positive"
    );
    for (a, name) in CLASSES.iter().enumerate() {
        let row = report.confusion[a];
        println!(
            "{:<18}{:>9}{:>9}{:>9}",
            format!("Actual {name}"),
            row[0],
            row[1],
            row[2]
        );
    }
    println!("\nWords that push hardest towards each class:");
    for (c, name) in CLASSES.iter().enumerate() {
        let weights = &model.models[c].weights;
        let mut order: Vec<usize> = (0..weights.len()).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
        let top: Vec<&str> = order
            .iter()
            .take(10)
            .map(|&i| model.vocab.words[i].as_str())
            .collect();
        println!(
            "
{name:<9} {}",
            top.join(", ")
        );
    }
    Ok(())
}
pub fn tune() -> AnyResult<()> {
    let tweets = text::load(text::TRAIN)?;
    let fold_of = text::assign_folds(&tweets);
    let lambdas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3];
    let epochs = 10;
    println!("5-fold cross-validation on the training tweets only\n");
    println!("{:>8} {:>9} {:>9}", "lambda", "accuracy", "macro F1");
    // Build each fold's vocabulary and vectors once, from that fold's training part only.
    let mut folds = Vec::new();
    for fold in 0..FOLDS {
        let fit_rows: Vec<&Tweet> = (0..tweets.len())
            .filter(|&i| fold_of[i] != fold)
            .map(|i| &tweets[i])
            .collect();
        let check_rows: Vec<&Tweet> = (0..tweets.len())
            .filter(|&i| fold_of[i] == fold)
            .map(|i| &tweets[i])
            .collect();
        let texts: Vec<&str> = fit_rows.iter().map(|t| t.text.as_str()).collect();
        let vocab = Vocabulary::fit(&texts);
        let x_fit: Vec<Sparse> = fit_rows.iter().map(|t| vocab.transform(&t.text)).collect();
        let y_fit: Vec<usize> = fit_rows.iter().map(|t| t.label).collect();
        let x_check: Vec<Sparse> = check_rows
            .iter()
            .map(|t| vocab.transform(&t.text))
            .collect();
        let y_check: Vec<usize> = check_rows.iter().map(|t| t.label).collect();
        folds.push((vocab.len(), x_fit, y_fit, x_check, y_check));
    }
    let mut best = (0.0, -1.0);
    for &lambda in &lambdas {
        let settings = SvmSettings { lambda, epochs };
        let mut actual = Vec::new();
        let mut predicted = Vec::new();
        for (dims, x_fit, y_fit, x_check, y_check) in &folds {
            let models = fit(x_fit, y_fit, *dims, settings, false);
            for (x, &y) in x_check.iter().zip(y_check) {
                let scores: Vec<f64> = models.iter().map(|m| m.score(x)).collect();
                predicted.push(argmax(&scores));
                actual.push(y);
            }
        }
        let report = score_predictions(&actual, &predicted);
        println!(
            "{lambda:>8} {:>8.2}% {:>9.3}",
            report.accuracy * 100.0,
            report.macro_f1
        );
        if report.macro_f1 > best.1 {
            best = (lambda, report.macro_f1);
        }
    }
    let settings = SvmSettings {
        lambda: best.0,
        epochs,
    };
    println!("\nBest: lambda {}, macro F1 {:.3}", best.0, best.1);
    let texts: Vec<&str> = tweets.iter().map(|t| t.text.as_str()).collect();
    let vocab = Vocabulary::fit(&texts);
    let xs = vectorize(&vocab, &tweets);
    let labels: Vec<usize> = tweets.iter().map(|t| t.label).collect();
    let models = fit(&xs, &labels, vocab.len(), settings, false);
    save_new_version(
        vocab,
        models,
        settings,
        tweets.len(),
        &format!("tuned: 5-fold macro F1 {:.3}", best.1),
    )?;
    println!("\nNow run: cargo run --release -- svm-evaluate");
    Ok(())
}
pub fn versions() -> AnyResult<()> {
    let catalog = load_catalog()?;
    if catalog.versions.is_empty() {
        println!("No SVM versions yet. Run: cargo run --release -- svm-train");
        return Ok(());
    }
    println!(
        "{:<8} {:>8} {:>7} {:>12}
{}",
        "version", "lambda", "passes", "trained", "note"
    );
    for v in &catalog.versions {
        let mark = if v.version == catalog.active {
            "*"
        } else {
            " "
        };
        println!(
            "{mark} v{:<5} {:>8} {:>7} {:>12}
{}",
            v.version, v.lambda, v.epochs, v.trained_unix, v.note
        );
    }
    println!("\n* = active (used by svm-evaluate and the server)");
    Ok(())
}
/// Roll back (or forward) to any saved version.
pub fn use_version() -> AnyResult<()> {
    let wanted: usize = std::env::args()
        .nth(2)
        .ok_or("usage: cargo run --release -- svm-use <version>")?
        .trim_start_matches('v')
        .parse()?;
    let mut catalog = load_catalog()?;
    if !catalog.versions.iter().any(|v| v.version == wanted) {
        return Err(format!("version {wanted} not found").into());
    }
    catalog.active = wanted;
    save_catalog(&catalog)?;
    println!("Active SVM version is now v{wanted}");
    Ok(())
}
// ---------- the same SVM on the heart data ----------
/// Scale each column with the training averages, then store rows as sparse vectors.
fn heart_vectors(x: &ndarray::Array2<f64>, means: &[f64], stds: &[f64]) -> Vec<Sparse> {
    x.rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| (j, (v - means[j]) / stds[j]))
                .collect()
        })
        .collect()
}
pub fn heart() -> AnyResult<()> {
    let train_df = data::load_csv(data::TRAIN)?;
    let features = data::feature_names(&train_df);
    let (x, y) = data::to_arrays(&train_df, &features)?;
    let labels: Vec<usize> = y.iter().map(|&b| b as usize).collect();
    let fold_of = data::assign_folds(&y);
    let dims = features.len();
    println!("Linear SVM on the Framingham data, 5-fold cross-validation\n");
    println!("{:>8} {:>7}", "lambda", "AUC");
    let mut best = (0.0, -1.0);
    for lambda in [1e-4, 1e-3, 1e-2, 1e-1] {
        let settings = SvmSettings {
            lambda,
            epochs: 200,
        };
        let mut scores = vec![0.0; labels.len()];
        for fold in 0..FOLDS {
            let fit_rows: Vec<usize> = (0..labels.len()).filter(|&i| fold_of[i] != fold).collect();
            let check_rows: Vec<usize> =
                (0..labels.len()).filter(|&i| fold_of[i] == fold).collect();
            let x_fit = x.select(ndarray::Axis(0), &fit_rows);
            let means: Vec<f64> = x_fit
                .columns()
                .into_iter()
                .map(|c| c.mean().unwrap())
                .collect();
            let stds: Vec<f64> = x_fit.columns().into_iter().map(|c| c.std(0.0)).collect();
            let xs_fit = heart_vectors(&x_fit, &means, &stds);
            let ys_fit: Vec<usize> = fit_rows.iter().map(|&i| labels[i]).collect();
            let models = fit(&xs_fit, &ys_fit, dims, settings, false);
            let xs_check = heart_vectors(&x.select(ndarray::Axis(0), &check_rows), &means, &stds);
            for (&row, xv) in check_rows.iter().zip(&xs_check) {
                scores[row] = models[1].score(xv);
            }
        }
        let a = auc(&scores, &y.to_vec());
        println!("{lambda:>8} {a:>7.3}");
        if a > best.1 {
            best = (lambda, a);
        }
    }
    // Final model on all training rows, scored once on the test set.
    let settings = SvmSettings {
        lambda: best.0,
        epochs: 200,
    };
    let means: Vec<f64> = x.columns().into_iter().map(|c| c.mean().unwrap()).collect();
    let stds: Vec<f64> = x.columns().into_iter().map(|c| c.std(0.0)).collect();
    let models = fit(
        &heart_vectors(&x, &means, &stds),
        &labels,
        dims,
        settings,
        false,
    );
    let test_df = data::load_csv(data::TEST)?;
    let (x_test, y_test) = data::to_arrays(&test_df, &features)?;
    let test_scores: Vec<f64> = heart_vectors(&x_test, &means, &stds)
        .iter()
        .map(|xv| models[1].score(xv))
        .collect();
    println!("\nBest: lambda {}", best.0);
    println!("Test AUC: {:.3}", auc(&test_scores, &y_test.to_vec()));
    println!("\nWeights (per 1 standard deviation), largest first:");
    let w = &models[1].weights;
    let mut order: Vec<usize> = (0..dims).collect();
    order.sort_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()));
    for &j in order.iter().take(6) {
        println!(
            "
{:<16} {:>8.4}",
            features[j], w[j]
        );
    }
    Ok(())
}
