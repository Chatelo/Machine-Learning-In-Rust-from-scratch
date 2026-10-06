# Machine Learning in Rust from Scratch

A chapter-by-chapter machine learning project using the Framingham heart study dataset, tweet text, Shakespeare text, and a real open-weight language model. This chapter implements Chapter 11: a Llama-style SmolLM2 model loaded and adapted with Candle.

## Chapter 11 — A real open model with Candle

This chapter swaps the tiny hand-written GPT from Chapter 10 for a real pretrained model: SmolLM2-135M, loaded from Hugging Face weights and run with Candle in Rust. The goal is not to rebuild every detail from first principles, but to show how an open model is loaded, compared against its reference implementation, and then tuned on Shakespeare.

### Check the model against Candle's reference

```bash
cargo run --release -- llm-check
```

This stage loads the pretrained model, inspects the vocabulary and tokenization, checks our implementation against Candle's own Llama forward pass, and confirms the key/value cache produces the same logits as a full recomputation.

### Write text from the pretrained model

```bash
cargo run --release -- llm-write "ROMEO:\n" 100 0.8
```

This stage writes new text from the original pretrained model. The optional final argument `tuned` swaps in the Shakespeare-fine-tuned checkpoint instead of the base model.

```bash
cargo run --release -- llm-write "ROMEO:\n" 100 0.8 tuned
```

### Score the model on held-back Shakespeare

```bash
cargo run --release -- llm-score
```

This stage compares the tiny GPT from Chapter 10 against the original and tuned SmolLM2 checkpoints on the same held-back Shakespeare text, reporting loss per character.

### Keep training the model on Shakespeare

```bash
cargo run --release -- llm-train
```

This stage loads the pretrained model, freezes most of the layers, trains only the top layers and final norm on the training split of Shakespeare, and saves the tuned model to `models/smollm2-shakespeare.safetensors`.

## Earlier chapters

- Chapter 1: raw-data exploration and summary statistics — [01-data-explore](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/01-data-explore)
- Chapter 2: cleaning and stratified train/test split — [02-data-cleaning-split](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/02-data-cleaning-split)
- Chapter 3: linear regression for `sysBP` — [03-linear-regression](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/03-linear-regression)
- Chapter 4: logistic regression, tuning, and HTTP prediction — [04-logistic-regression-for-classification](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/04-logistic-regression-for-classification)
- Chapter 5: decision tree, tuning, and explainable prediction — [05-decision-trees](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/05-decision-trees)
- Chapter 6: random forest, OOB tuning, and POST /predict/forest — [06-random-forest](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/06-random-forest)
- Chapter 7: text preprocessing, SVMs, and tweet sentiment — [07-text-svm](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/07-text-svm)
- Chapter 8: Naive Bayes and k-nearest neighbours — [08-naive-bayes-knn](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/08-naive-bayes-knn)
- Chapter 9: neural networks — [09-neural-networks](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/09-neural-networks)
- Chapter 10: tiny GPT — [10-tiny-gpt](https://github.com/Chatelo/Machine-Learning-In-Rust-from-scratch/tree/10-tiny-gpt)
- Current chapter: Chapter 11 — SmolLM2 with Candle

## Project files

- [src/main.rs](src/main.rs) — stage selector for the chapter commands, including `llm-check`, `llm-write`, `llm-score`, and `llm-train`
- [src/llm.rs](src/llm.rs) — Candle model loader, RoPE attention, KV cache, and continued training loop
- [src/gpt.rs](src/gpt.rs) — Chapter 10 tiny GPT reference implementation used for comparison
- [src/server.rs](src/server.rs) — HTTP endpoint for generating text with the tuned LLM
- [models/smollm2-135m](models/smollm2-135m) — pretrained model files, tokenizer, and config
- [models/smollm2-shakespeare.safetensors](models/smollm2-shakespeare.safetensors) — Shakespeare-tuned weights
- [data/shakespeare.txt](data/shakespeare.txt) — text used for the continued training and evaluation

The first build may take a few minutes while Cargo compiles dependencies; later runs are much faster.
