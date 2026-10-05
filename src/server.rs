//! Serving predictions over HTTP.
//!
//!POST /predictlogistic regression (Chapter 4)
//!POST /predict/treedecision tree (Chapter 5)
//!POST /predict/forest random forest (Chapter 6)
use crate::AnyResult;
use crate::forest::{self, SavedForest};
use crate::logistic::{self, SavedLogisticModel};
use crate::tree::{self, SavedTree};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use ndarray::Array2;
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};
const ADDRESS: &str = "127.0.0.1:3000";
/// Both models, loaded once and shared by every request.
struct AppState {
    logistic: SavedLogisticModel,
    tree: SavedTree,
    forest: SavedForest,
}
type ApiError = (StatusCode, String);
#[derive(Serialize)]
pub struct Prediction {
    model: &'static str,
    probability: f64,
    threshold: f64,
    at_risk: bool,
    /// Only the tree can say WHY: the answers it followed to reach its leaf.
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<Vec<String>>,
    /// Forest only: the share of trees that, on their own, say "at risk".
    #[serde(skip_serializing_if = "Option::is_none")]
    trees_agreeing: Option<f64>,
}
pub fn serve() -> AnyResult<()> {
    let logistic: SavedLogisticModel =
        serde_json::from_str(&std::fs::read_to_string(logistic::MODEL)?)?;
    let tree: SavedTree = serde_json::from_str(&std::fs::read_to_string(tree::MODEL)?)?;
    let forest: SavedForest = serde_json::from_str(&std::fs::read_to_string(forest::MODEL)?)?;
    println!(
        "Loaded {} (alpha {}, threshold {:.2})",
        logistic::MODEL,
        logistic.alpha,
        logistic.threshold
    );
    println!(
        "Loaded {} (depth {}, min leaf {}, threshold {:.2})",
        tree::MODEL,
        tree.settings.max_depth,
        tree.settings.min_leaf,
        tree.threshold
    );
    println!(
        "Loaded {} ({} trees, {} features per split, threshold {:.2})",
        forest::MODEL,
        forest.settings.n_trees,
        forest.settings.max_features,
        forest.threshold
    );
    println!("Expects: {}", logistic.features.join(", "));
    let app = Router::new()
        .route("/predict", post(predict_logistic))
        .route("/predict/tree", post(predict_tree))
        .route("/predict/forest", post(predict_forest))
        .with_state(Arc::new(AppState {
            logistic,
            tree,
            forest,
        }));
    // main() is not async, so start the async runtime here.
    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind(ADDRESS).await?;
        println!("Listening on http://{ADDRESS}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}
/// Put the values into the exact order the model was trained with,
/// or list everything that is missing or not a real number.
fn read_row(features: &[String], input: &HashMap<String, f64>) -> Result<Vec<f64>, ApiError> {
    let mut row = Vec::with_capacity(features.len());
    let mut missing = Vec::new();
    for name in features {
        match input.get(name) {
            Some(value) if value.is_finite() => row.push(*value),
            _ => missing.push(name.as_str()),
        }
    }
    if missing.is_empty() {
        Ok(row)
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            format!("Missing or invalid: {}\n", missing.join(", ")),
        ))
    }
}
async fn predict_logistic(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<Prediction>, ApiError> {
    let model = &state.logistic;
    let row = read_row(&model.features, &input)?;
    let x = Array2::from_shape_vec((1, row.len()), row)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let probability = logistic::predict_probs(model, &x)[0];
    Ok(Json(Prediction {
        model: "logistic",
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
        path: None,
        trees_agreeing: None,
    }))
}
async fn predict_tree(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<Prediction>, ApiError> {
    let model = &state.tree;
    let row = read_row(&model.features, &input)?;
    let (probability, path) = tree::explain(&model.root, &row, &model.features);
    Ok(Json(Prediction {
        model: "tree",
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
        path: Some(path),
        trees_agreeing: None,
    }))
}
async fn predict_forest(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<Prediction>, ApiError> {
    let model = &state.forest;
    let row = read_row(&model.features, &input)?;
    let x = Array2::from_shape_vec((1, row.len()), row)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let probability = forest::predict_probs(&model.trees, &x)[0];
    let agreeing = forest::agreement(&model.trees, x.row(0), model.threshold);
    Ok(Json(Prediction {
        model: "forest",
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
        path: None,
        trees_agreeing: Some(agreeing),
    }))
}
