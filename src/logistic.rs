//! Chapter 4: logistic regression.

use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::AnyResult;
use crate::data::{FOLDS, TARGET, TEST, TRAIN, assign_folds, feature_names, load_csv, to_arrays};
use crate::metrics::{Confusion, auc, report};

pub const MODEL: &str = "logistic_model.json";

/// Everything needed to make a prediction later:
/// the feature order, the scaling numbers, and the learned weights.

#[derive(Serialize, Deserialize)]
pub struct SavedLogisticModel {
    pub target: String,
    pub features: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub alpha: f64,
    pub intercept: f64,
    pub weights: Vec<f64>,
    pub threshold: f64,
}

// ---------- model helpers (used by train, tune and evaluate) ----------
/// Scale with this data's own averages/spreads, fit linfa, and return
/// everything needed to predict later.

pub fn fit_model(
    features: &[String],
    x: &Array2<f64>,
    y: &Array1<bool>,
    alpha: f64,
    threshold: f64,
) -> AnyResult<SavedLogisticModel> {
    let means: Vec<f64> = x.columns().into_iter().map(|c| c.mean().unwrap()).collect();
    let stds: Vec<f64> = x.columns().into_iter().map(|c| c.std(0.0)).collect();

    let mut scaled = x.clone();
    for (j, mut column) in scaled.columns_mut().into_iter().enumerate() {
        column.mapv_inplace(|v| (v - means[j]) / stds[j]);
    }

    let dataset = Dataset::new(scaled, y.clone());
    let fitted = LogisticRegression::default()
        .alpha(alpha)
        .max_iterations(200)
        .fit(&dataset)?;

    // linfa learns weights for whichever class is MORE common.
    // Here that is class 0, so flip the signs to get weights for class 1.

    let sign = if fitted.labels().pos.class { 1.0 } else { -1.0 };
    Ok(SavedLogisticModel {
        target: TARGET.to_string(),
        features: features.to_vec(),
        means,
        stds,
        alpha,
        intercept: sign * fitted.intercept(),
        weights: fitted.params().iter().map(|w| sign * w).collect(),
        threshold,
    })
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Scale with the SAVED numbers, add up weight x value,
/// then squeeze into 0..1 with the sigmoid.

pub fn predict_probs(model: &SavedLogisticModel, x: &Array2<f64>) -> Vec<f64> {
    x.rows()
        .into_iter()
        .map(|row| {
            let total = row
                .iter()
                .enumerate()
                .map(|(j, v)| model.weights[j] * (v - model.means[j]) / model.stds[j])
                .sum::<f64>()
                + model.intercept;
            sigmoid(total)
        })
        .collect()
}

// ---------- stage 3: train ----------

pub fn train() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let features = feature_names(&df);
    let (x, y) = to_arrays(&df, &features)?;
    // linfa's default settings: alpha 1.0, threshold 0.5.
    let model = fit_model(&features, &x, &y, 1.0, 0.5)?;

    println!("Intercept: {:.4}", model.intercept);
    println!("\nWeights (per 1 standard deviation):");
    for (name, weight) in features.iter().zip(&model.weights) {
        println!(
            "
    {name:<16} {weight:>8.4}"
        );
    }
    std::fs::write(MODEL, serde_json::to_string_pretty(&model)?)?;
    println!("\nSaved: {MODEL}");
    Ok(())
}

// ---------- stage 4: evaluate ----------

pub fn evaluate() -> AnyResult<()> {
    let model: SavedLogisticModel = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let df = load_csv(TEST)?;
    let (x, y) = to_arrays(&df, &model.features)?;
    let probs = predict_probs(&model, &x);
    let actual: Vec<bool> = y.to_vec();
    let baseline = actual.iter().filter(|&&a| !a).count() as f64 / actual.len() as f64;
    println!("Model alpha: {}", model.alpha);
    println!("Test rows: {}", actual.len());
    println!("Always-predict-0 accuracy: {:.2}%", baseline * 100.0);
    println!("AUC: {:.3}", auc(&probs, &actual));
    report(0.5, &probs, &actual);
    if model.threshold != 0.5 {
        report(model.threshold, &probs, &actual);
    }
    Ok(())
}

// ---------- stage 5: tune ----------
pub fn tune() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let features = feature_names(&df);
    let (x, y) = to_arrays(&df, &features)?;
    let actual: Vec<bool> = y.to_vec();
    let fold_of = assign_folds(&y);
    let alphas = [0.1, 1.0, 10.0, 100.0, 1000.0];
    let thresholds: Vec<f64> = (1..=10).map(|i| i as f64 * 5.0 / 100.0).collect(); // 0.05 .. 0.50
    let mut best = (0.0, 0.0, -1.0); // (alpha, threshold, f1)
    println!("5-fold cross-validation on the training data only\n");
    println!("{:>8} {:>7} {:>10} {:>7}", "alpha", "AUC", "best cut", "F1");
    for &alpha in &alphas {
        // Every training row gets a probability from a model that did NOT see it.
        let mut probs = vec![0.0; actual.len()];
        for fold in 0..FOLDS {
            let fit_rows: Vec<usize> = (0..actual.len()).filter(|&i| fold_of[i] != fold).collect();
            let check_rows: Vec<usize> =
                (0..actual.len()).filter(|&i| fold_of[i] == fold).collect();
            let x_fit = x.select(ndarray::Axis(0), &fit_rows);
            let y_fit = y.select(ndarray::Axis(0), &fit_rows);
            let x_check = x.select(ndarray::Axis(0), &check_rows);
            let model = fit_model(&features, &x_fit, &y_fit, alpha, 0.5)?;
            for (&row, p) in check_rows.iter().zip(predict_probs(&model, &x_check)) {
                probs[row] = p;
            }
        }
        // Best threshold for this alpha, judged by F1.
        let (cut, f1) = thresholds
            .iter()
            .map(|&t| (t, Confusion::new(t, &probs, &actual).f1()))
            .fold(
                (0.0, -1.0),
                |acc, cur| if cur.1 > acc.1 { cur } else { acc },
            );
        println!(
            "{alpha:>8} {:>7.3} {cut:>10.2} {f1:>7.3}",
            auc(&probs, &actual)
        );
        if f1 > best.2 {
            best = (alpha, cut, f1);
        }
    }
    let (alpha, threshold, f1) = best;
    println!("\nBest: alpha {alpha}, threshold {threshold:.2}, F1 {f1:.3}");
    // Final model: all training rows, best settings.
    let model = fit_model(&features, &x, &y, alpha, threshold)?;
    std::fs::write(MODEL, serde_json::to_string_pretty(&model)?)?;
    println!("Saved: {MODEL}");
    println!("\nNow run: cargo run -- evaluate");
    Ok(())
}
