//! Serving predictions over HTTP.
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use ndarray::Array2;
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};

use crate::AnyResult;
use crate::logistic::{MODEL, SavedLogisticModel, predict_probs};

const ADDRESS: &str = "127.0.0.1:3000";

#[derive(Serialize)]
pub struct Prediction {
    probability: f64,
    threshold: f64,
    at_risk: bool,
}

pub fn serve() -> AnyResult<()> {
    let model: SavedLogisticModel = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;

    println!(
        "Loaded {MODEL} (alpha {}, threshold {:.2})",
        model.alpha, model.threshold
    );
    println!("Expects: {}", model.features.join(", "));

    let app = Router::new()
        .route("/predict", post(predict))
        .with_state(Arc::new(model));

    // main() is not async, so start the async runtime here.
    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind(ADDRESS).await?;
        println!("Listening on http://{ADDRESS}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}

pub async fn predict(
    State(model): State<Arc<SavedLogisticModel>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<Prediction>, (StatusCode, String)> {
    // Put the values into the exact order the model was trained with.
    let mut row = Vec::with_capacity(model.features.len());
    let mut missing = Vec::new();

    for name in &model.features {
        match input.get(name) {
            Some(value) if value.is_finite() => row.push(*value),
            _ => missing.push(name.as_str()),
        }
    }

    if !missing.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Missing or invalid: {}\n", missing.join(", ")),
        ));
    }

    let x = Array2::from_shape_vec((1, row.len()), row)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    // let probability = predict_probs(&model, &x)[1];
    let probability = predict_probs(&model, &x)[0];

    Ok(Json(Prediction {
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
    }))
}
