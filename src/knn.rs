//! Chapter 8: k-nearest neighbours.
//!
//! No training at all: keep every training patient. To judge a new patient,
//! find the k most similar ones and report the share who developed heart
//! disease. "Similar" means close together once every feature is scaled.
use crate::AnyResult;
use crate::data::{self, FOLDS};
use crate::metrics::{auc, best_threshold, report};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
pub const MODEL: &str = "knn_model.json";
const KS: [usize; 7] = [1, 5, 15, 25, 51, 101, 201];
/// The whole training set is the model.
#[derive(Serialize, Deserialize)]
pub struct SavedKnn {
    pub features: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub k: usize,
    pub threshold: f64,
    pub rows: Vec<Vec<f64>>, // raw values, as in the CSV
    pub labels: Vec<bool>,
}
fn column_stats(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len() as f64;
    let dims = rows[0].len();
    let means: Vec<f64> = (0..dims)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n)
        .collect();
    let stds: Vec<f64> = (0..dims)
        .map(|j| (rows.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / n).sqrt())
        .collect();
    (means, stds)
}
fn scale(rows: &[Vec<f64>], means: &[f64], stds: &[f64]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(j, v)| (v - means[j]) / stds[j])
                .collect()
        })
        .collect()
}
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
/// For each query, the `max_k` closest reference rows as (row, distance), closest first.
pub fn nearest(
    reference: &[Vec<f64>],
    queries: &[Vec<f64>],
    max_k: usize,
) -> Vec<Vec<(usize, f64)>> {
    queries
        .par_iter()
        .map(|q| {
            let mut d: Vec<(usize, f64)> = reference
                .iter()
                .enumerate()
                .map(|(i, r)| (i, distance(q, r)))
                .collect();
            let k = max_k.min(d.len());
            // Move the k closest to the front without sorting everything, then sort just those.
            d.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
            d.truncate(k);
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            d
        })
        .collect()
}
/// Share of the first k neighbours who developed heart disease.
fn share(neighbours: &[(usize, f64)], labels: &[bool], k: usize) -> f64 {
    neighbours
        .iter()
        .take(k)
        .filter(|(i, _)| labels[*i])
        .count() as f64
        / k as f64
}
fn rows_of(x: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    x.rows().into_iter().map(|r| r.to_vec()).collect()
}
pub fn tune() -> AnyResult<()> {
    let df = data::load_csv(data::TRAIN)?;
    let features = data::feature_names(&df);
    let (x, y) = data::to_arrays(&df, &features)?;
    let rows = rows_of(&x);
    let labels: Vec<bool> = y.to_vec();
    let fold_of = data::assign_folds(&y);
    let max_k = *KS.last().unwrap();
    // Out-of-fold probabilities for every k, with and without scaling.
    let mut probs_scaled = vec![vec![0.0; rows.len()]; KS.len()];
    let mut probs_raw = vec![vec![0.0; rows.len()]; KS.len()];
    for fold in 0..FOLDS {
        let fit: Vec<usize> = (0..rows.len()).filter(|&i| fold_of[i] != fold).collect();
        let check: Vec<usize> = (0..rows.len()).filter(|&i| fold_of[i] == fold).collect();
        let fit_rows: Vec<Vec<f64>> = fit.iter().map(|&i| rows[i].clone()).collect();
        let check_rows: Vec<Vec<f64>> = check.iter().map(|&i| rows[i].clone()).collect();
        let fit_labels: Vec<bool> = fit.iter().map(|&i| labels[i]).collect();
        let (means, stds) = column_stats(&fit_rows);
        let near_scaled = nearest(
            &scale(&fit_rows, &means, &stds),
            &scale(&check_rows, &means, &stds),
            max_k,
        );
        let near_raw = nearest(&fit_rows, &check_rows, max_k);
        for (q, &row) in check.iter().enumerate() {
            for (ki, &k) in KS.iter().enumerate() {
                probs_scaled[ki][row] = share(&near_scaled[q], &fit_labels, k);
                probs_raw[ki][row] = share(&near_raw[q], &fit_labels, k);
            }
        }
    }
    println!("k-nearest neighbours, 5-fold cross-validation on the training data\n");
    println!("{:>5} {:>12} {:>14}", "k", "AUC scaled", "AUC unscaled");
    let mut best = (0, -1.0);
    for (ki, &k) in KS.iter().enumerate() {
        let a_scaled = auc(&probs_scaled[ki], &labels);
        println!(
            "{k:>5} {a_scaled:>12.3} {:>14.3}",
            auc(&probs_raw[ki], &labels)
        );
        if a_scaled > best.1 {
            best = (ki, a_scaled);
        }
    }
    let k = KS[best.0];
    let (threshold, f1) = best_threshold(&probs_scaled[best.0], &labels);
    println!("\nBest: k {k} (scaled), cut-off {threshold:.2}, F1 {f1:.3}");
    let (means, stds) = column_stats(&rows);
    let model = SavedKnn {
        features,
        means,
        stds,
        k,
        threshold,
        rows,
        labels,
    };
    std::fs::write(MODEL, serde_json::to_string(&model)?)?;
    println!(
        "Saved: {MODEL} (the model is the {} training patients)",
        model.rows.len()
    );
    Ok(())
}
impl SavedKnn {
    /// Probability and the k nearest training patients for each query row (raw values).
    pub fn predict(&self, queries: &[Vec<f64>]) -> Vec<(f64, Vec<(usize, f64)>)> {
        let reference = scale(&self.rows, &self.means, &self.stds);
        let near = nearest(&reference, &scale(queries, &self.means, &self.stds), self.k);
        near.into_iter()
            .map(|n| (share(&n, &self.labels, self.k), n))
            .collect()
    }
}
pub fn evaluate() -> AnyResult<()> {
    let model: SavedKnn = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let df = data::load_csv(data::TEST)?;
    let (x, y) = data::to_arrays(&df, &model.features)?;
    let probs: Vec<f64> = model
        .predict(&rows_of(&x))
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let actual: Vec<bool> = y.to_vec();
    println!("k-NN: k {}, {} stored patients", model.k, model.rows.len());
    println!("Test AUC: {:.3}", auc(&probs, &actual));
    report(model.threshold, &probs, &actual);
    Ok(())
}
