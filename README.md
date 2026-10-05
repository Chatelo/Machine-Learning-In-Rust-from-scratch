# Machine Learning in Rust from Scratch

A chapter-by-chapter machine learning project using the Framingham heart study dataset. This branch implements Chapter 5: a handwritten decision tree with tuning and a prediction API.

## Chapter 5 — Decision tree, tuning, and POST /predict/tree

This chapter trains a decision tree that predicts whether a person is at risk of heart disease. The tree stores a probability in each leaf, uses greedy Gini-split rules to grow itself, and exposes the exact decision path used for each prediction.

### Train the tree

```bash
cargo run --release -- tree-train
```

The training script reads [data/framingham_train.csv](data/framingham_train.csv), grows a depth-limited tree with a minimum leaf-size guardrail, prints the rule tree, counts the leaves, and saves the model to `tree_model.json`.

### Evaluate the tree

```bash
cargo run --release -- tree-evaluate
```

The evaluation stage loads the saved tree, scores it on [data/framingham_test.csv](data/framingham_test.csv), prints AUC and thresholded classification metrics, and compares the model against the baseline of predicting zero all the time.

### Tune the tree

```bash
cargo run --release -- tree-tune
```

This runs 5-fold cross-validation on the training set, explores several depth and minimum-leaf settings, and picks the best threshold by F1 score before saving the tuned model.

### Serve the prediction API

```bash
cargo run --release -- serve
```

The HTTP server exposes:

- `POST /predict` for the logistic model
- `POST /predict/tree` for the decision tree

The tree endpoint is explainable: it returns both the predicted probability and the rule path that led to the leaf decision.

## Earlier chapters

- Chapter 1: raw-data exploration and summary statistics — [01-data-explore](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/01-data-explore)
- Chapter 2: cleaning and stratified train/test split — [02-data-cleaning-split](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/02-data-cleaning-split)
- Chapter 3: linear regression for `sysBP` — [03-linear-regression](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/03-linear-regression)
- Chapter 4: logistic regression, tuning, and HTTP prediction — [04-logistic-regression-for-classification](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/04-logistic-regression-for-classification)
- Current branch: Chapter 5 — [05-decision-trees](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/05-decision-trees)

## Project files

- [src/main.rs](src/main.rs) — stage selector for `explore`, `clean`, `split`, `linear-train`, `linear-evaluate`, `train`, `evaluate`, `tune`, `serve`, `tree-train`, `tree-evaluate`, and `tree-tune`
- [src/data.rs](src/data.rs) — shared loading, cleaning, splitting, and cross-validation helpers
- [src/explore.rs](src/explore.rs) — chapter 1 exploration
- [src/linear.rs](src/linear.rs) — chapter 3 linear-regression workflow
- [src/logistic.rs](src/logistic.rs) — chapter 4 logistic-regression workflow
- [src/tree.rs](src/tree.rs) — chapter 5 decision-tree training, evaluation, tuning, and explanation logic
- [src/server.rs](src/server.rs) — HTTP prediction endpoint for both the logistic and tree models
- [data/framingham_train.csv](data/framingham_train.csv) — training split
- [data/framingham_test.csv](data/framingham_test.csv) — test split

The first build may take a few minutes while Cargo compiles dependencies; later runs are much faster.
