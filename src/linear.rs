//! Chapter 3: linear regression, predicting a number.
//!
//! Predict a person's systolic blood pressure (sysBP, in mmHg) from eight
//! everyday measurements, with ordinary least squares:
//!
//! predicted sysBP = intercept + w1·male + w2·age + ... + w8·diabetes
//!
//! Least squares picks the weights that make the squared errors on the
//! training people as small as possible.

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::AnyResult;
use crate::data::{TEST, TRAIN, load_csv};

pub const MODEL: &str = "linear_model.json";
const TARGET: &str = "sysBP";

/// The inputs. Left out on purpose: diaBP, prevalentHyp and BPMeds, which are
/// themselves blood-pressure measurements. Using them would hand the model the answer.

const FEATURES: [&str; 8] = [
    "male",
    "age",
    "cigsPerDay",
    "BMI",
    "totChol",
    "glucose",
    "heartRate",
    "diabetes",
];

#[derive(Serialize, Deserialize)]
pub struct SavedLinearModel {
    pub target: String,
    pub features: Vec<String>,
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    /// The average sysBP in training: The "no model" baseline guess.
    pub train_mean: f64,
}

impl SavedLinearModel {
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        self.intercept
            + row
                .iter()
                .zip(&self.coefficients)
                .map(|(x, w)| x * w)
                .sum::<f64>()
    }
}

fn column(df: &DataFrame, name: &str) -> AnyResult<Vec<f64>> {
    let values = df.column(name)?.cast(&DataType::Float64)?;
    Ok(values.f64()?.into_no_null_iter().collect())
}

/// Feature grid (people × features) and the target list.
fn arrays(df: &DataFrame) -> AnyResult<(Array2<f64>, Array1<f64>)> {
    let mut x = Array2::<f64>::zeros((df.height(), FEATURES.len()));

    for (j, name) in FEATURES.iter().enumerate() {
        for (i, v) in column(df, name)?.into_iter().enumerate() {
            x[[i, j]] = v;
        }
    }

    let y = Array1::from(column(df, TARGET)?);
    Ok((x, y))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Pearson correlation: +1 rises together, −1 one rises as the other falls, 0 no straight-line link.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// Mean absolute error, root mean squared error, and R² (share of the variation explained).

fn errors(predicted: &[f64], actual: &[f64]) -> (f64, f64, f64) {
    let n = actual.len() as f64;
    let mae = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / n;
    let sse: f64 = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| (p - a).powi(2))
        .sum();
    let m = mean(actual);
    let sst: f64 = actual.iter().map(|a| (a - m).powi(2)).sum();
    (mae, (sse / n).sqrt(), 1.0 - sse / sst)
}

pub fn train() -> AnyResult<()> {
    let df = load_csv(TRAIN)?;
    let (x, y) = arrays(&df)?;
    let target = y.to_vec();

    println!(
        "Predicting {TARGET} (mmHg) from {} features, {} training people\n",
        FEATURES.len(),
        df.height()
    );

    // Check first that the inputs carry information about the target.

    println!("Correlation of each feature with {TARGET} (-1 to 1):");
    for (j, name) in FEATURES.iter().enumerate() {
        println!(
            "  {name:<11} {:>6.3}",
            correlation(&x.column(j).to_vec(), &target)
        );
    }

    let fitted = LinearRegression::new().fit(&Dataset::new(x.clone(), y))?;
    let model = SavedLinearModel {
        target: TARGET.to_string(),
        features: FEATURES.iter().map(|s| s.to_string()).collect(),
        intercept: fitted.intercept(),
        coefficients: fitted.params().to_vec(),
        train_mean: mean(&target),
    };

    println!("\nIntercept: {:.2} mmHg", model.intercept);
    println!("Each extra 1 of a feature changes predicted {TARGET} by:");

    for (name, w) in model.features.iter().zip(&model.coefficients) {
        println!(" {name:<11} {w:+8.3} mmHg");
    }

    let predicted: Vec<f64> = x
        .rows()
        .into_iter()
        .map(|row| model.predict_one(row.as_slice().unwrap()))
        .collect();
    let (mae, rmse, r2) = errors(&predicted, &target);
    println!("\nOn the training people: MAE {mae:.2}, RMSE {rmse:.2}, R² {r2:.3}");

    std::fs::write(MODEL, serde_json::to_string_pretty(&model)?)?;

    println!("Saved: {MODEL}");

    Ok(())
}

pub fn evaluate() -> AnyResult<()> {
    let model: SavedLinearModel = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let df = load_csv(TEST)?;
    let (x, y) = arrays(&df)?;
    let actual = y.to_vec();

    let predicted: Vec<f64> = x
        .rows()
        .into_iter()
        .map(|row| model.predict_one(row.as_slice().unwrap()))
        .collect();
    let baseline = vec![model.train_mean; actual.len()];
    let (mae, rmse, r2) = errors(&predicted, &actual);
    let (b_mae, b_rmse, b_r2) = errors(&baseline, &actual);
    println!("Test set: {} people\n", actual.len());
    println!("{:<28} {:>7} {:>7} {:>7}", "", "MAE", "RMSE", "R²");
    println!(
        "{:<28} {:>7.2} {:>7.2} {:>7.3}",
        "Always guess the average", b_mae, b_rmse, b_r2
    );
    println!(
        "{:<28} {:>7.2} {:>7.2} {:>7.3}",
        "Linear regression", mae, rmse, r2
    );
    println!("\nFirst five test people:");
    println!(
        "{:>6} {:>5} {:>10} {:>10}",
        "age", "male", "actual", "predicted"
    );

    for i in 0..5 {
        println!(
            "{:>6} {:>5} {:>10.1} {:>10.1}",
            x[[i, 1]],
            x[[i, 0]],
            actual[i],
            predicted[i]
        );
    }

    let within = |limit: f64| {
        predicted
            .iter()
            .zip(&actual)
            .filter(|(p, a)| (*p - *a).abs() <= limit)
            .count() as f64
            / actual.len() as f64
            * 100.0
    };

    println!(
        "\nPredictions within 10 mmHg: {:.1}% within 20 mmHg: {:.1}%",
        within(10.0),
        within(20.0)
    );
    Ok(())
}
