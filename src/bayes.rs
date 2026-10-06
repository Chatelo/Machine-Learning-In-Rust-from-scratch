//! Chapter 8: Naive Bayes, two ways.
//!
//! Bayes' rule:P(class | data) is proportional to P(class) × P(data | class)
//! "Naive": treat every feature as independent, so P(data | class) is just
//! the product of one probability per feature.
//!
//! Gaussian Naive Bayes for the heart numbers (each feature is a bell curve),
//! Multinomial Naive Bayes for tweet words (each word is a count).
use crate::AnyResult;
use crate::data::{self, FOLDS};
use crate::logistic;
use crate::metrics::{auc, best_threshold, calibration, report};
use crate::text::{self, CLASSES, Sparse, Tweet, Vocabulary};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::time::Instant;
// ================= Gaussian Naive Bayes (heart data) =================
/// For each class: how common it is, and each feature's average and spread.
#[derive(Serialize, Deserialize)]
pub struct GaussianNb {
    pub features: Vec<String>,
    pub log_prior: [f64; 2],
    pub means: [Vec<f64>; 2],
    pub vars: [Vec<f64>; 2],
    pub threshold: f64,
}
impl GaussianNb {
    pub fn fit(features: &[String], x: &Array2<f64>, y: &Array1<bool>) -> Self {
        let mut log_prior = [0.0; 2];
        let mut means: [Vec<f64>; 2] = [vec![], vec![]];
        let mut vars: [Vec<f64>; 2] = [vec![], vec![]];
        // A tiny extra spread so no feature has a variance of exactly zero.
        let biggest_var = x
            .columns()
            .into_iter()
            .map(|c| c.var(0.0))
            .fold(0.0, f64::max);
        let epsilon = 1e-9 * biggest_var;
        for class in [false, true] {
            let rows: Vec<usize> = (0..y.len()).filter(|&i| y[i] == class).collect();
            let part = x.select(ndarray::Axis(0), &rows);
            let c = class as usize;
            log_prior[c] = (rows.len() as f64 / y.len() as f64).ln();
            means[c] = part
                .columns()
                .into_iter()
                .map(|col| col.mean().unwrap())
                .collect();
            vars[c] = part
                .columns()
                .into_iter()
                .map(|col| col.var(0.0) + epsilon)
                .collect();
        }
        GaussianNb {
            features: features.to_vec(),
            log_prior,
            means,
            vars,
            threshold: 0.5,
        }
    }
    /// Log of P(class) × P(each feature | class), added up (logs turn × into +).
    fn log_score(&self, row: ArrayView1<f64>, c: usize) -> f64 {
        let mut total = self.log_prior[c];
        for (j, &value) in row.iter().enumerate() {
            let (mean, var) = (self.means[c][j], self.vars[c][j]);
            // log of the bell-curve height at this value
            total += -0.5 * (2.0 * std::f64::consts::PI * var).ln()
                - (value - mean).powi(2) / (2.0 * var);
        }
        total
    }
    pub fn predict_probs(&self, x: &Array2<f64>) -> Vec<f64> {
        x.rows()
            .into_iter()
            .map(|row| {
                let (s0, s1) = (self.log_score(row, 0), self.log_score(row, 1));
                // P(class 1) = e^s1 / (e^s0 + e^s1), computed safely
                1.0 / (1.0 + (s0 - s1).exp())
            })
            .collect()
    }
}
pub fn heart() -> AnyResult<()> {
    let df = data::load_csv(data::TRAIN)?;
    let features = data::feature_names(&df);
    let (x, y) = data::to_arrays(&df, &features)?;
    let actual: Vec<bool> = y.to_vec();
    let fold_of = data::assign_folds(&y);
    // Out-of-fold probabilities for Naive Bayes AND logistic regression,
    // so both are judged on training rows they never saw.
    let mut nb_probs = vec![0.0; actual.len()];
    let mut lr_probs = vec![0.0; actual.len()];
    for fold in 0..FOLDS {
        let fit_rows: Vec<usize> = (0..actual.len()).filter(|&i| fold_of[i] != fold).collect();
        let check_rows: Vec<usize> = (0..actual.len()).filter(|&i| fold_of[i] == fold).collect();
        let x_fit = x.select(ndarray::Axis(0), &fit_rows);
        let y_fit = y.select(ndarray::Axis(0), &fit_rows);
        let x_check = x.select(ndarray::Axis(0), &check_rows);
        let nb = GaussianNb::fit(&features, &x_fit, &y_fit);
        let lr = logistic::fit_model(&features, &x_fit, &y_fit, 0.1, 0.5)?;
        for ((&row, p_nb), p_lr) in check_rows
            .iter()
            .zip(nb.predict_probs(&x_check))
            .zip(logistic::predict_probs(&lr, &x_check))
        {
            nb_probs[row] = p_nb;
            lr_probs[row] = p_lr;
        }
    }
    println!("Gaussian Naive Bayes, 5-fold cross-validation on the training data");
    println!("Cross-validated AUC: {:.3}", auc(&nb_probs, &actual));
    let (threshold, f1) = best_threshold(&nb_probs, &actual);
    println!("Best cut-off: {threshold:.2} (F1 {f1:.3})");
    println!("\nDo the probabilities mean what they say? (training rows, out-of-fold)\n");
    println!("Naive Bayes:");
    calibration(&nb_probs, &actual);
    println!("\nLogistic regression, same folds:");
    calibration(&lr_probs, &actual);
    // Final model on all training rows, tested once.
    let mut model = GaussianNb::fit(&features, &x, &y);
    model.threshold = threshold;
    let test_df = data::load_csv(data::TEST)?;
    let (x_test, y_test) = data::to_arrays(&test_df, &features)?;
    let probs = model.predict_probs(&x_test);
    let test_actual: Vec<bool> = y_test.to_vec();
    println!("\nTest AUC: {:.3}", auc(&probs, &test_actual));
    report(threshold, &probs, &test_actual);
    std::fs::write("nb_model.json", serde_json::to_string_pretty(&model)?)?;
    println!("\nSaved: nb_model.json");
    Ok(())
}
// ================= Multinomial Naive Bayes (tweets) =================
/// For each class: how common it is, and how likely each word is.
pub struct MultinomialNb {
    log_prior: Vec<f64>,
    log_word: Vec<Vec<f64>>, // [class][word]
}
impl MultinomialNb {
    /// Count words per class. `alpha` pretends every word was seen `alpha`
    /// extra times in every class, so an unseen word never scores zero.
    pub fn fit(xs: &[Sparse], labels: &[usize], n_words: usize, alpha: f64) -> Self {
        let n_classes = CLASSES.len();
        let mut docs = vec![0.0; n_classes];
        let mut counts = vec![vec![alpha; n_words]; n_classes];
        for (x, &c) in xs.iter().zip(labels) {
            docs[c] += 1.0;
            for &(i, n) in x {
                counts[c][i] += n;
            }
        }
        let log_prior = docs.iter().map(|d| (d / xs.len() as f64).ln()).collect();
        let log_word = counts
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum();
                row.iter().map(|n| (n / total).ln()).collect()
            })
            .collect();
        MultinomialNb {
            log_prior,
            log_word,
        }
    }
    pub fn predict(&self, x: &Sparse) -> usize {
        let scores: Vec<f64> = (0..self.log_prior.len())
            .map(|c| {
                self.log_prior[c] + x.iter().map(|&(i, n)| n * self.log_word[c][i]).sum::<f64>()
            })
            .collect();
        crate::svm::argmax(&scores)
    }
}
pub fn tweets() -> AnyResult<()> {
    let tweets = text::load(text::TRAIN)?;
    let fold_of = text::assign_folds(&tweets);
    let alphas = [0.1, 0.3, 1.0, 3.0];
    println!("Multinomial Naive Bayes on tweets, 5-fold cross-validation\n");
    println!("{:>6} {:>9} {:>9}", "alpha", "accuracy", "macro F1");
    // Each fold gets its own word list, as in Chapter 7.
    let mut folds = Vec::new();
    for fold in 0..FOLDS {
        let fit: Vec<&Tweet> = (0..tweets.len())
            .filter(|&i| fold_of[i] != fold)
            .map(|i| &tweets[i])
            .collect();
        let check: Vec<&Tweet> = (0..tweets.len())
            .filter(|&i| fold_of[i] == fold)
            .map(|i| &tweets[i])
            .collect();
        let vocab = Vocabulary::fit(&fit.iter().map(|t| t.text.as_str()).collect::<Vec<_>>());
        let x_fit: Vec<Sparse> = fit.iter().map(|t| vocab.counts(&t.text)).collect();
        let y_fit: Vec<usize> = fit.iter().map(|t| t.label).collect();
        let x_check: Vec<Sparse> = check.iter().map(|t| vocab.counts(&t.text)).collect();
        let y_check: Vec<usize> = check.iter().map(|t| t.label).collect();
        folds.push((vocab.len(), x_fit, y_fit, x_check, y_check));
    }
    let mut best = (0.0, -1.0);
    for &alpha in &alphas {
        let mut actual = Vec::new();
        let mut predicted = Vec::new();
        for (n_words, x_fit, y_fit, x_check, y_check) in &folds {
            let model = MultinomialNb::fit(x_fit, y_fit, *n_words, alpha);
            predicted.extend(x_check.iter().map(|x| model.predict(x)));
            actual.extend_from_slice(y_check);
        }
        let r = crate::svm::score_predictions(&actual, &predicted);
        println!(
            "{alpha:>6} {:>8.2}% {:>9.3}",
            r.accuracy * 100.0,
            r.macro_f1
        );
        if r.macro_f1 > best.1 {
            best = (alpha, r.macro_f1);
        }
    }
    println!("\nBest: alpha {}", best.0);
    // Final model, timed, tested once.
    let vocab = Vocabulary::fit(&tweets.iter().map(|t| t.text.as_str()).collect::<Vec<_>>());
    let xs: Vec<Sparse> = tweets.iter().map(|t| vocab.counts(&t.text)).collect();
    let labels: Vec<usize> = tweets.iter().map(|t| t.label).collect();
    let start = Instant::now();
    let model = MultinomialNb::fit(&xs, &labels, vocab.len(), best.0);
    println!(
        "Training time: {:.1} ms (it is just counting)",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let test = text::load(text::TEST)?;
    let actual: Vec<usize> = test.iter().map(|t| t.label).collect();
    let predicted: Vec<usize> = test
        .iter()
        .map(|t| model.predict(&vocab.counts(&t.text)))
        .collect();
    let r = crate::svm::score_predictions(&actual, &predicted);
    println!("\nTest accuracy: {:.2}%", r.accuracy * 100.0);
    println!("Test macro F1: {:.3}", r.macro_f1);
    // Words most typical of each class compared with the other two.
    // Only words used at least 30 times: rarer ones give noisy ratios.
    let mut used = vec![0.0; vocab.len()];
    for x in &xs {
        for &(i, n) in x {
            used[i] += n;
        }
    }
    println!("\nWords most typical of each class (used 30+ times):");
    for (c, name) in CLASSES.iter().enumerate() {
        let mut ratio: Vec<(usize, f64)> = (0..vocab.len())
            .filter(|&i| used[i] >= 30.0)
            .map(|i| {
                let others = (0..CLASSES.len())
                    .filter(|&o| o != c)
                    .map(|o| model.log_word[o][i])
                    .fold(f64::MIN, f64::max);
                (i, model.log_word[c][i] - others)
            })
            .collect();
        ratio.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<&str> = ratio
            .iter()
            .take(10)
            .map(|(i, _)| vocab.words[*i].as_str())
            .collect();
        println!(
            "
{name:<9} {}",
            top.join(", ")
        );
    }
    Ok(())
}
