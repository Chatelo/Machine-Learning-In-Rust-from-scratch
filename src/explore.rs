//! Chapter 1: a first look at the data.
//!
//! Before any model, answer four questions: how big is the table, what is in
//! each column, what is missing, and how common is the outcome we want to predict?
use polars::prelude::*;
use crate::AnyResult;

const RAW: &str = "data/framingham.csv";


pub fn explore() -> AnyResult<()> {
    // Read the CSV into a DataFrame: a table with named, typed columns.
    // "NA" in the file means "no value recorded", so read it as missing.
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(None) //look at every raw before choosing colum type
        .map_parse_options(|opts| opts.with_null_values(Some(NullValues::AllColumnsSingle("NA".into()))))
        .try_into_reader_with_file_path(Some(RAW.into()))?
        .finish()?;

    println!("Rows: {}, Columns: {}\n", df.height(), df.width());
    println!("First 5 rows: \n{}\n", df.head(Some(5)));

    println!("{:<16} {:<6} {:>8}", "column", "type", "missing");

    for col in df.columns() {
        println!("{:>16} {:<6} {:>8}", col.name().as_str(), col.dtype().to_string(), col.null_count());
    }

    // The outcome: did the person develop coronary heart disease within 10 years?
    let ones = df
        .column("TenYearCHD")?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .filter(|&v| v==1)
        .count();
    println!(
        "\nTenYearCHD = 1 (heart disease within 10 years): {ones} of {} people ({:.1}%)",
        df.height(),
        ones as f64 / df.height() as f64 * 100.0
    );
    
    println!("\n{:<12} {:>8} {:>8} {:>8}", "column", "min", "mean", "max");

    for name in ["age", "sysBP", "totChol", "BMI", "cigsPerDay", "glucose"]{
        let values = df.column(name)?.cast(&DataType::Float64)?;
        let values = values.f64()?;

        println!(
            "{:<12} {:>8.1} {:>8.1} {:>8.1}",
            name,
            values.min().unwrap_or(f64::NAN),
            values.mean().unwrap_or(f64::NAN),
            values.max().unwrap_or(f64::NAN) 
        );
    }
    Ok(())

}