//! Loading, cleaning and splitting the Framingham data.
//! Shared by every model.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::AnyResult;

pub const RAW: &str = "data/framingham.csv";
pub const CLEAN: &str = "data/framingham_clean.csv";
pub const TRAIN: &str = "data/framingham_train.csv";
pub const TEST: &str = "data/framingham_test.csv";

pub const TARGET: &str = "TenYearCHD";
pub const SEED: u64 = 42;
pub const TEST_SHARE: f64 = 0.2;
pub const FOLDS: usize = 5;

pub fn load_csv(path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(None)
        .map_parse_options(|opts| {
            opts.with_null_values(Some(NullValues::AllColumnsSingle("NA".into())))
        })
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
}

pub fn save_csv(df: &mut DataFrame, path: &str) -> PolarsResult<()> {
    let mut file = std::fs::File::create(path)?;

    CsvWriter::new(&mut file).finish(df)
}

pub fn class_counts(df: &DataFrame) -> PolarsResult<(usize, usize)> {
    let ones = df
        .column(TARGET)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .filter(|&v| v == 1)
        .count();

    Ok((df.height() - ones, ones))
}

pub fn print_balance(label: &str, df: &DataFrame) -> PolarsResult<()> {
    let (zeros, ones) = class_counts(df)?;

    let total = df.height() as f64;

    println!(
        "{label}: {} rows | class 0: {zeros} ({:.2}%) | class 1: {ones} ({:.2}%)",
        df.height(),
        zeros as f64 / total * 100.0,
        ones as f64 / total * 100.0
    );
    Ok(())
}

pub fn feature_names(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .into_iter()
        .filter(|name| name.as_str() != TARGET)
        .map(|name| name.to_string())
        .collect()
}

/// Turn a table into a feature grid (rows x features) and a target list.

pub fn to_arrays(df: &DataFrame, features: &[String]) -> AnyResult<(Array2<f64>, Array1<bool>)> {
    let mut x = Array2::<f64>::zeros((df.height(), features.len()));

    for (j, name) in features.iter().enumerate() {
        let column = df.column(name)?.cast(&DataType::Float64)?;

        for (i, value) in column.f64()?.into_no_null_iter().enumerate() {
            x[[i, j]] = value;
        }
    }

    let y: Array1<bool> = df
        .column(TARGET)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .map(|value| value == 1)
        .collect();

    Ok((x, y))
}

// ----------------Stage 1: Clean---------------------------

pub fn clean() -> AnyResult<()> {
    let raw = load_csv(RAW)?;
    print_balance("Raw", &raw)?;

    println!("\nMissing values per column");
    for col in raw.columns() {
        if col.null_count() > 0 {
            println!(" {:<12} {}", col.name(), col.null_count());
        }
    }

    // `education` is not a health measurement, and it has missing values.
    let mut cleaned = raw.drop("education")?.drop_nulls::<String>(None)?;

    println!();

    print_balance("Cleared", &cleaned)?;

    save_csv(&mut cleaned, CLEAN)?;
    println!("Saved: {CLEAN}");

    Ok(())
}

// ---------- stage 2: split ----------

pub fn split() -> AnyResult<()> {
    let df = load_csv(CLEAN)?;
    let target: Vec<i64> = df
        .column(TARGET)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .collect();
    // Shuffle each class separately, then take 20% of each for testing.
    // This keeps the class mix the same in both parts.
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut train_rows: Vec<IdxSize> = Vec::new();
    let mut test_rows: Vec<IdxSize> = Vec::new();
    for class in [0, 1] {
        let mut rows: Vec<IdxSize> = target
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value == class)
            .map(|(i, _)| i as IdxSize)
            .collect();
        rows.shuffle(&mut rng);
        let n_test = (rows.len() as f64 * TEST_SHARE).round() as usize;
        test_rows.extend_from_slice(&rows[..n_test]);
        train_rows.extend_from_slice(&rows[n_test..]);
    }
    // Mix the two classes together again inside each part.
    train_rows.shuffle(&mut rng);
    test_rows.shuffle(&mut rng);
    let mut train = df.take(&IdxCa::from_vec("idx".into(), train_rows))?;
    let mut test = df.take(&IdxCa::from_vec("idx".into(), test_rows))?;
    print_balance("Train", &train)?;
    print_balance("Test", &test);

    save_csv(&mut train, TRAIN)?;
    save_csv(&mut test, TEST)?;

    println!("Saved: {TRAIN}");
    println!("Saved: {TEST}");

    Ok(())
}

// ---------- cross-validation folds ----------
/// Give every training row a fold number 0..FOLDS,
/// dealing each class out separately so every fold has the same class mix.

pub fn assign_folds(y: &Array1<bool>) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut fold_of = vec![0; y.len()];

    for class in [false, true] {
        let mut rows: Vec<usize> = (0..y.len()).filter(|&i| y[i] == class).collect();
        rows.shuffle(&mut rng);

        for (k, &row) in rows.iter().enumerate() {
            fold_of[row] = k % FOLDS;
        }
    }
    fold_of
}
