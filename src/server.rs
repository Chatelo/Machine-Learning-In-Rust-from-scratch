//! Serving predictions over HTTP.
//!
//!POST /predictlogistic regression (Chapter 4)
//!POST /predict/treedecision tree (Chapter 5)
//!POST /predict/forest random forest (Chapter 6)
//!POST /predict/sentiment
//!POST /predict/knnk-nearest neighbours, with the most similar patients (Chapter 8)
//!POST /predict/nnneural network (Chapter 9) tweet sentiment, linear SVM (Chapter 7)
use crate::AnyResult;
use crate::forest::{self, SavedForest};
use crate::knn::{self, SavedKnn};
use crate::logistic::{self, SavedLogisticModel};
use crate::nn::{self, SavedHeartNet};
use crate::svm::{self, SavedSvm};
use crate::tree::{self, SavedTree};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
const ADDRESS: &str = "127.0.0.1:3000";
/// Both models, loaded once and shared by every request.
struct AppState {
    logistic: SavedLogisticModel,
    tree: SavedTree,
    forest: SavedForest,
    svm: SavedSvm,
    knn: SavedKnn,
    nn: SavedHeartNet,
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
    let svm = svm::load_active()?;
    let knn: SavedKnn = serde_json::from_str(&std::fs::read_to_string(knn::MODEL)?)?;
    let nn: SavedHeartNet = serde_json::from_str(&std::fs::read_to_string(nn::HEART_MODEL)?)?;
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
    println!(
        "Loaded SVM v{} ({} words, lambda {})",
        svm.version,
        svm.vocab.len(),
        svm.settings.lambda
    );
    println!(
        "Loaded {} (k {}, {} stored patients)",
        knn::MODEL,
        knn.k,
        knn.rows.len()
    );
    println!(
        "Loaded {} ({} hidden units, threshold {:.2})",
        nn::HEART_MODEL,
        nn.net.layers[0].b.len(),
        nn.threshold
    );
    println!("Expects: {}", logistic.features.join(", "));
    let app = Router::new()
        .route("/predict", post(predict_logistic))
        .route("/predict/tree", post(predict_tree))
        .route("/predict/forest", post(predict_forest))
        .route("/predict/sentiment", post(predict_sentiment))
        .route("/predict/knn", post(predict_knn))
        .route("/predict/nn", post(predict_nn))
        .with_state(Arc::new(AppState {
            logistic,
            tree,
            forest,
            svm,
            knn,
            nn,
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
#[derive(Deserialize)]
struct SentimentRequest {
    text: String,
}
#[derive(Serialize)]
struct SentimentReply {
    model: &'static str,
    version: usize,
    sentiment: String,
    scores: BTreeMap<String, f64>,
    /// The words in the text that pushed hardest towards the answer.
    because_of: Vec<String>,
}
async fn predict_sentiment(
    State(state): State<Arc<AppState>>,
    Json(input): Json<SentimentRequest>,
) -> Result<Json<SentimentReply>, ApiError> {
    let model = &state.svm;
    if input.text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text is empty\n".to_string()));
    }
    let (class, scores) = model.predict(&input.text);
    Ok(Json(SentimentReply {
        model: "svm",
        version: model.version,
        sentiment: model.classes[class].clone(),
        scores: model.classes.iter().cloned().zip(scores).collect(),
        because_of: model.top_words(&input.text, class, 5),
    }))
}
/// One of the most similar training patients, shown by a few key features.
#[derive(Serialize)]
struct Neighbour {
    distance: f64,
    features: BTreeMap<String, f64>,
    developed_chd: bool,
}
#[derive(Serialize)]
struct KnnReply {
    model: &'static str,
    probability: f64,
    threshold: f64,
    at_risk: bool,
    k: usize,
    closest: Vec<Neighbour>,
}
const SHOWN: [&str; 5] = ["age", "male", "sysBP", "totChol", "cigsPerDay"];
async fn predict_knn(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<KnnReply>, ApiError> {
    let model = &state.knn;
    let row = read_row(&model.features, &input)?;
    let (probability, neighbours) = model.predict(&[row]).remove(0);
    let closest = neighbours
        .iter()
        .take(5)
        .map(|&(i, distance)| Neighbour {
            distance,
            features: model
                .features
                .iter()
                .zip(&model.rows[i])
                .filter(|(name, _)| SHOWN.contains(&name.as_str()))
                .map(|(name, v)| (name.clone(), *v))
                .collect(),
            developed_chd: model.labels[i],
        })
        .collect();
    Ok(Json(KnnReply {
        model: "knn",
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
        k: model.k,
        closest,
    }))
}
async fn predict_nn(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, f64>>,
) -> Result<Json<Prediction>, ApiError> {
    let model = &state.nn;
    let row = read_row(&model.features, &input)?;
    let x = Array2::from_shape_vec((1, row.len()), row)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let probability = model.probs(&x)[0];
    Ok(Json(Prediction {
        model: "nn",
        probability,
        threshold: model.threshold,
        at_risk: probability >= model.threshold,
        path: None,
        trees_agreeing: None,
    }))
}
