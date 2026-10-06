//! Chapter 10: a tiny GPT, written from scratch.
//!
//! A character-level transformer: it reads text one character at a time
//! and learns to predict the next one. Every forward and backward pass is
//! written by hand, like `nn.rs`, and proved with a gradient check.
//!
//!characters → embeddings (+ position) → [attention → MLP] × layers → next-character probabilities
use crate::AnyResult;
use crate::data::SEED;
use ndarray::{Array1, Array2, Axis, s};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::time::Instant;
pub const TEXT: &str = "data/shakespeare.txt";
pub const MODEL: &str = "gpt_model.json";
// ================= characters =================
/// The model's alphabet: every distinct character in the text, sorted.
#[derive(Serialize, Deserialize, Clone)]
pub struct Tokenizer {
    pub chars: Vec<char>,
}
impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        Tokenizer { chars }
    }
    /// Text → token ids. Characters the model has never seen are skipped.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.chars.binary_search(&c).ok())
            .collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}
/// The first 90% of the text for training, the last 10% held back.
pub fn load() -> AnyResult<(Tokenizer, Vec<usize>, Vec<usize>)> {
    let text = std::fs::read_to_string(TEXT)?;
    let tok = Tokenizer::from_text(&text);
    let ids = tok.encode(&text);
    let cut = ids.len() * 9 / 10;
    Ok((tok, ids[..cut].to_vec(), ids[cut..].to_vec()))
}
// ================= model shape and parameters =================
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub vocab: usize,
    // number of distinct characters
    pub context: usize, // how many characters the model can look back over
    pub dim: usize,     // size of each character's vector
    pub heads: usize,   // attention heads per layer
    pub layers: usize,  // transformer blocks
}
/// One transformer block: attention, then a small MLP, each with LayerNorm first.
#[derive(Serialize, Deserialize, Clone)]
pub struct Block {
    ln1_g: Array1<f64>,
    ln1_b: Array1<f64>,
    w_qkv: Array2<f64>, // dim × 3·dim: queries, keys and values in one matrix
    b_qkv: Array1<f64>,
    w_o: Array2<f64>,
    // dim × dim: mixes the heads' outputs
    b_o: Array1<f64>,
    ln2_g: Array1<f64>,
    ln2_b: Array1<f64>,
    w_1: Array2<f64>,
    // dim × 4·dim
    b_1: Array1<f64>,
    w_2: Array2<f64>,
    // 4·dim × dim
    b_2: Array1<f64>,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct Gpt {
    pub config: Config,
    tok_emb: Array2<f64>, // vocab × dim: one learned vector per character
    pos_emb: Array2<f64>, // context × dim: one learned vector per position
    blocks: Vec<Block>,
    lnf_g: Array1<f64>,
    lnf_b: Array1<f64>,
    head: Array2<f64>,
    // dim × vocab: vector → a score for every character
    head_b: Array1<f64>,
}
/// A normally distributed random number (Box–Muller), as in `nn.rs`.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
impl Gpt {
    pub fn new(config: Config, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = config.dim;
        let mut rand =
            |r: usize, k: usize| Array2::from_shape_fn((r, k), |_| gaussian(&mut rng) * 0.02);
        let blocks = (0..config.layers)
            .map(|_| Block {
                ln1_g: Array1::ones(c),
                ln1_b: Array1::zeros(c),
                w_qkv: rand(c, 3 * c),
                b_qkv: Array1::zeros(3 * c),
                w_o: rand(c, c),
                b_o: Array1::zeros(c),
                ln2_g: Array1::ones(c),
                ln2_b: Array1::zeros(c),
                w_1: rand(c, 4 * c),
                b_1: Array1::zeros(4 * c),
                w_2: rand(4 * c, c),
                b_2: Array1::zeros(c),
            })
            .collect();
        Gpt {
            config,
            tok_emb: rand(config.vocab, c),
            pos_emb: rand(config.context, c),
            blocks,
            lnf_g: Array1::ones(c),
            lnf_b: Array1::zeros(c),
            head: rand(c, config.vocab),
            head_b: Array1::zeros(config.vocab),
        }
    }
    /// Every parameter as a flat slice, always in the same order (for Adam and the gradient check).
    fn slices_mut(&mut self) -> Vec<&mut [f64]> {
        let mut v: Vec<&mut [f64]> = vec![
            self.tok_emb.as_slice_mut().unwrap(),
            self.pos_emb.as_slice_mut().unwrap(),
        ];
        for b in self.blocks.iter_mut() {
            v.extend([
                b.ln1_g.as_slice_mut().unwrap(),
                b.ln1_b.as_slice_mut().unwrap(),
                b.w_qkv.as_slice_mut().unwrap(),
                b.b_qkv.as_slice_mut().unwrap(),
                b.w_o.as_slice_mut().unwrap(),
                b.b_o.as_slice_mut().unwrap(),
                b.ln2_g.as_slice_mut().unwrap(),
                b.ln2_b.as_slice_mut().unwrap(),
                b.w_1.as_slice_mut().unwrap(),
                b.b_1.as_slice_mut().unwrap(),
                b.w_2.as_slice_mut().unwrap(),
                b.b_2.as_slice_mut().unwrap(),
            ]);
        }
        v.extend([
            self.lnf_g.as_slice_mut().unwrap(),
            self.lnf_b.as_slice_mut().unwrap(),
            self.head.as_slice_mut().unwrap(),
            self.head_b.as_slice_mut().unwrap(),
        ]);
        v
    }
    fn zeros_like(&self) -> Gpt {
        let mut g = self.clone();
        for s in g.slices_mut() {
            s.fill(0.0);
        }
        g
    }
    pub fn n_params(&self) -> usize {
        self.clone().slices_mut().iter().map(|s| s.len()).sum()
    }
}
// ================= building blocks: forward and backward =================
struct NormCache {
    xhat: Array2<f64>,
    inv_std: Array1<f64>,
}
/// LayerNorm: rescale each row to average 0 and spread 1, then apply a learned gain and shift.
fn layer_norm(x: &Array2<f64>, g: &Array1<f64>, b: &Array1<f64>) -> (Array2<f64>, NormCache) {
    let n = x.ncols() as f64;
    let mut xhat = x.clone();
    let mut inv_std = Array1::zeros(x.nrows());
    for (i, mut row) in xhat.rows_mut().into_iter().enumerate() {
        let mean = row.sum() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        row.mapv_inplace(|v| (v - mean) * inv);
        inv_std[i] = inv;
    }
    let out = &xhat * g + b;
    (out, NormCache { xhat, inv_std })
}
fn layer_norm_back(
    dout: &Array2<f64>,
    g: &Array1<f64>,
    cache: &NormCache,
    dg: &mut Array1<f64>,
    db: &mut Array1<f64>,
) -> Array2<f64> {
    *dg += &(dout * &cache.xhat).sum_axis(Axis(0));
    *db += &dout.sum_axis(Axis(0));
    let n = dout.ncols() as f64;
    let dxhat = dout * g;
    let mut dx = Array2::zeros(dout.raw_dim());
    for i in 0..dout.nrows() {
        let (dh, xh) = (dxhat.row(i), cache.xhat.row(i));
        let sum_dh = dh.sum();
        let sum_dh_xh = (&dh * &xh).sum();
        let inv = cache.inv_std[i];
        dx.row_mut(i)
            .assign(&((&dh * n - sum_dh - &xh * sum_dh_xh) * (inv / n)));
    }
    dx
}
fn softmax_rows(z: &mut Array2<f64>) {
    for mut row in z.rows_mut() {
        let max = row.fold(f64::MIN, |m, &v| m.max(v));
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row /= sum;
    }
}
/// What the forward pass remembers for the backward pass.
struct BlockCache {
    x_in: Array2<f64>,
    ln1_out: Array2<f64>,
    ln1: NormCache,
    qkv: Array2<f64>,
    probs: Vec<Array2<f64>>, // attention weights, one T×T matrix per (sequence, head)
    att_out: Array2<f64>,
    // heads' outputs side by side, before w_o
    ln2_out: Array2<f64>,
    ln2: NormCache,
    hidden: Array2<f64>,
}
struct Cache {
    ids: Vec<usize>,
    batch: usize,
    t: usize,
    blocks: Vec<BlockCache>,
    lnf_out: Array2<f64>,
    lnf: NormCache,
    probs: Array2<f64>, // next-character probabilities, one row per position
}
impl Gpt {
    /// Forward pass over `batch` sequences of `t` characters each (ids laid out row after row).
    fn forward(&self, ids: &[usize], batch: usize, t: usize) -> Cache {
        let cfg = self.config;
        let (c, h) = (cfg.dim, cfg.heads);
        let hd = c / h;
        let scale = 1.0 / (hd as f64).sqrt();
        // 1. Embeddings: each character's vector plus its position's vector.
        let mut x = Array2::zeros((batch * t, c));
        for (n, &id) in ids.iter().enumerate() {
            let pos = n % t;
            x.row_mut(n)
                .assign(&(&self.tok_emb.row(id) + &self.pos_emb.row(pos)));
        }
        let mut caches = Vec::with_capacity(cfg.layers);
        for blk in &self.blocks {
            let x_in = x.clone();
            // 2. Attention: each position gathers information from earlier positions.
            let (ln1_out, ln1) = layer_norm(&x, &blk.ln1_g, &blk.ln1_b);
            let qkv = ln1_out.dot(&blk.w_qkv) + &blk.b_qkv;
            let mut att_out = Array2::zeros((batch * t, c));
            let mut probs = Vec::with_capacity(batch * h);
            for bi in 0..batch {
                let rows = bi * t..(bi + 1) * t;
                for hi in 0..h {
                    let q = qkv.slice(s![rows.clone(), hi * hd..(hi + 1) * hd]);
                    let k = qkv.slice(s![rows.clone(), c + hi * hd..c + (hi + 1) * hd]);
                    let v = qkv.slice(s![rows.clone(), 2 * c + hi * hd..2 * c + (hi + 1) * hd]);
                    // How well does each query match each key?
                    let mut scores = q.dot(&k.t()) * scale;
                    // Causal mask: position i may not look at later positions.
                    for i in 0..t {
                        for j in i + 1..t {
                            scores[[i, j]] = f64::NEG_INFINITY;
                        }
                    }
                    softmax_rows(&mut scores);
                    att_out
                        .slice_mut(s![rows.clone(), hi * hd..(hi + 1) * hd])
                        .assign(&scores.dot(&v));
                    probs.push(scores);
                }
            }
            x = &x + &(att_out.dot(&blk.w_o) + &blk.b_o); // residual connection
            // 3. MLP: each position thinks about what it gathered.
            let (ln2_out, ln2) = layer_norm(&x, &blk.ln2_g, &blk.ln2_b);
            let hidden = (ln2_out.dot(&blk.w_1) + &blk.b_1).mapv(|v| v.max(0.0));
            x = &x + &(hidden.dot(&blk.w_2) + &blk.b_2); // residual connection
            caches.push(BlockCache {
                x_in,
                ln1_out,
                ln1,
                qkv,
                probs,
                att_out,
                ln2_out,
                ln2,
                hidden,
            });
        }
        // 4. Scores for every possible next character, turned into probabilities.
        let (lnf_out, lnf) = layer_norm(&x, &self.lnf_g, &self.lnf_b);
        let mut probs = lnf_out.dot(&self.head) + &self.head_b;
        softmax_rows(&mut probs);
        Cache {
            ids: ids.to_vec(),
            batch,
            t,
            blocks: caches,
            lnf_out,
            lnf,
            probs,
        }
    }
    /// Average cross-entropy of predicting `targets` (the next character at every position).
    fn loss_of(cache: &Cache, targets: &[usize]) -> f64 {
        -targets
            .iter()
            .enumerate()
            .map(|(n, &y)| cache.probs[[n, y]].max(1e-12).ln())
            .sum::<f64>()
            / targets.len() as f64
    }
    pub fn loss(&self, ids: &[usize], targets: &[usize], batch: usize, t: usize) -> f64 {
        Gpt::loss_of(&self.forward(ids, batch, t), targets)
    }
    /// Backpropagation through the whole model. Returns (loss, gradients).
    fn backward(&self, ids: &[usize], targets: &[usize], batch: usize, t: usize) -> (f64, Gpt) {
        let cache = self.forward(ids, batch, t);
        let loss = Gpt::loss_of(&cache, targets);
        let cfg = self.config;
        let (c, h) = (cfg.dim, cfg.heads);
        let hd = c / h;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut g = self.zeros_like();
        // Output: softmax + cross-entropy → (probability − 1 at the right character) / n
        let n = targets.len() as f64;
        let mut dlogits = cache.probs.clone();
        for (i, &y) in targets.iter().enumerate() {
            dlogits[[i, y]] -= 1.0;
        }
        dlogits /= n;
        g.head = cache.lnf_out.t().dot(&dlogits);
        g.head_b = dlogits.sum_axis(Axis(0));
        let d_lnf = dlogits.dot(&self.head.t());
        let mut dx = layer_norm_back(&d_lnf, &self.lnf_g, &cache.lnf, &mut g.lnf_g, &mut g.lnf_b);
        for (l, blk) in self.blocks.iter().enumerate().rev() {
            let bc = &cache.blocks[l];
            let gb = &mut g.blocks[l];
            // MLP (the residual passes dx straight through as well)
            gb.w_2 = bc.hidden.t().dot(&dx);
            gb.b_2 = dx.sum_axis(Axis(0));
            let mut dhidden = dx.dot(&blk.w_2.t());
            dhidden.zip_mut_with(&bc.hidden, |d, &a| {
                if a <= 0.0 {
                    *d = 0.0
                }
            });
            gb.w_1 = bc.ln2_out.t().dot(&dhidden);
            gb.b_1 = dhidden.sum_axis(Axis(0));
            let d_ln2 = dhidden.dot(&blk.w_1.t());
            dx = dx + layer_norm_back(&d_ln2, &blk.ln2_g, &bc.ln2, &mut gb.ln2_g, &mut gb.ln2_b);
            // Attention output projection
            gb.w_o = bc.att_out.t().dot(&dx);
            gb.b_o = dx.sum_axis(Axis(0));
            let d_att = dx.dot(&blk.w_o.t());
            // Each head, each sequence
            let mut d_qkv = Array2::zeros(bc.qkv.raw_dim());
            for bi in 0..batch {
                let rows = bi * t..(bi + 1) * t;
                for hi in 0..h {
                    let p = &bc.probs[bi * h + hi];
                    let q = bc.qkv.slice(s![rows.clone(), hi * hd..(hi + 1) * hd]);
                    let k = bc
                        .qkv
                        .slice(s![rows.clone(), c + hi * hd..c + (hi + 1) * hd]);
                    let v = bc
                        .qkv
                        .slice(s![rows.clone(), 2 * c + hi * hd..2 * c + (hi + 1) * hd]);
                    let dy = d_att.slice(s![rows.clone(), hi * hd..(hi + 1) * hd]);
                    let dv = p.t().dot(&dy);
                    let dp = dy.dot(&v.t());
                    // Softmax backward, row by row: ds = p ⊙ (dp − Σ dp·p)
                    let mut ds = Array2::zeros((t, t));
                    for i in 0..t {
                        let dot: f64 = (0..t).map(|j| dp[[i, j]] * p[[i, j]]).sum();
                        for j in 0..t {
                            ds[[i, j]] = p[[i, j]] * (dp[[i, j]] - dot) * scale;
                        }
                    }
                    let dq = ds.dot(&k);
                    let dk = ds.t().dot(&q);
                    d_qkv
                        .slice_mut(s![rows.clone(), hi * hd..(hi + 1) * hd])
                        .assign(&dq);
                    d_qkv
                        .slice_mut(s![rows.clone(), c + hi * hd..c + (hi + 1) * hd])
                        .assign(&dk);
                    d_qkv
                        .slice_mut(s![rows.clone(), 2 * c + hi * hd..2 * c + (hi + 1) * hd])
                        .assign(&dv);
                }
            }
            gb.w_qkv = bc.ln1_out.t().dot(&d_qkv);
            gb.b_qkv = d_qkv.sum_axis(Axis(0));
            let d_ln1 = d_qkv.dot(&blk.w_qkv.t());
            dx = dx + layer_norm_back(&d_ln1, &blk.ln1_g, &bc.ln1, &mut gb.ln1_g, &mut gb.ln1_b);
            let _ = &bc.x_in;
        }
        // Embeddings: each row's gradient goes to its character and its position.
        for (row, &id) in cache.ids.iter().enumerate() {
            let pos = row % cache.t;
            let d = dx.row(row).to_owned();
            let mut te = g.tok_emb.row_mut(id);
            te += &d;
            let mut pe = g.pos_emb.row_mut(pos);
            pe += &d;
        }
        let _ = cache.batch;
        (loss, g)
    }
}
// ================= training =================
struct Adam {
    lr: f64,
    t: i32,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}
impl Adam {
    fn new(model: &Gpt, lr: f64) -> Self {
        let shapes: Vec<Vec<f64>> = model
            .clone()
            .slices_mut()
            .iter()
            .map(|s| vec![0.0; s.len()])
            .collect();
        Adam {
            lr,
            t: 0,
            m: shapes.clone(),
            v: shapes,
        }
    }
    fn step(&mut self, model: &mut Gpt, grads: &mut Gpt) {
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        self.t += 1;
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for (((p, g), m), v) in model
            .slices_mut()
            .into_iter()
            .zip(grads.slices_mut())
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            for i in 0..p.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                p[i] -= self.lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
            }
        }
    }
}
/// `batch` random windows of `t` characters, and the character after each position.
fn get_batch(data: &[usize], batch: usize, t: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
    let mut x = Vec::with_capacity(batch * t);
    let mut y = Vec::with_capacity(batch * t);
    for _ in 0..batch {
        let start = rng.random_range(0..data.len() - t - 1);
        x.extend_from_slice(&data[start..start + t]);
        y.extend_from_slice(&data[start + 1..start + t + 1]);
    }
    (x, y)
}
/// Average loss over the same fixed windows every time, so numbers are comparable.
fn estimate_loss(model: &Gpt, data: &[usize], batches: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let t = model.config.context;
    (0..batches)
        .map(|_| {
            let (x, y) = get_batch(data, 16, t, &mut rng);
            model.loss(&x, &y, 16, t)
        })
        .sum::<f64>()
        / batches as f64
}
/// Write `n` new characters after `prompt`, one at a time.
pub fn generate(
    model: &Gpt,
    tok: &Tokenizer,
    prompt: &str,
    n: usize,
    temperature: f64,
    seed: u64,
) -> String {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids = tok.encode(prompt);
    if ids.is_empty() {
        ids.push(tok.encode("\n")[0]);
    }
    for _ in 0..n {
        let start = ids.len().saturating_sub(model.config.context);
        let window = &ids[start..];
        let cache = model.forward(window, 1, window.len());
        let last = cache.probs.row(window.len() - 1);
        // Temperature: < 1 makes likely characters even likelier; > 1 flattens the choice.
        let weights: Vec<f64> = last
            .iter()
            .map(|p| p.max(1e-12).powf(1.0 / temperature))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut r = rng.random::<f64>() * total;
        let mut next = weights.len() - 1;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                next = i;
                break;
            }
            r -= w;
        }
        ids.push(next);
    }
    tok.decode(&ids)
}
#[derive(Serialize, Deserialize)]
pub struct SavedGpt {
    pub tokenizer: Tokenizer,
    pub model: Gpt,
}
// ================= stages =================
pub fn data() -> AnyResult<()> {
    let (tok, train, val) = load()?;
    println!(
        "Characters: {} training, {} held back",
        train.len(),
        val.len()
    );
    println!(
        "Alphabet ({} characters): {:?}",
        tok.chars.len(),
        tok.chars.iter().collect::<String>()
    );
    println!("First 40 ids: {:?}", &train[..40]);
    println!("Decoded back: {:?}", tok.decode(&train[..40]));
    // Baseline 1: guess uniformly.
    let v = tok.chars.len();
    println!(
        "\nLoss if every character is equally likely: {:.3}",
        (v as f64).ln()
    );
    // Baseline 2: a bigram model, which only looks at the previous character (it is just counting).
    let mut counts = vec![vec![1.0; v]; v];
    for w in train.windows(2) {
        counts[w[0]][w[1]] += 1.0;
    }
    let probs: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().sum();
            row.iter().map(|c| c / total).collect()
        })
        .collect();
    let val_loss =
        -val.windows(2).map(|w| probs[w[0]][w[1]].ln()).sum::<f64>() / (val.len() - 1) as f64;
    println!("Bigram model (previous character only), held-back loss: {val_loss:.3}");
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut ids = vec![tok.encode("\n")[0]];
    for _ in 0..200 {
        let row = &probs[*ids.last().unwrap()];
        let mut r = rng.random::<f64>();
        let mut next = v - 1;
        for (i, p) in row.iter().enumerate() {
            if r < *p {
                next = i;
                break;
            }
            r -= p;
        }
        ids.push(next);
    }
    println!("\nBigram sample:\n{}", tok.decode(&ids));
    Ok(())
}
/// Compare backprop with finite differences on a tiny model.
pub fn check() -> AnyResult<()> {
    let (tok, train, _) = load()?;
    let cfg = Config {
        vocab: tok.chars.len(),
        context: 8,
        dim: 8,
        heads: 2,
        layers: 2,
    };
    let model = Gpt::new(cfg, SEED);
    let mut rng = StdRng::seed_from_u64(SEED);
    let (x, y) = get_batch(&train, 2, 8, &mut rng);
    let (_, mut grads) = model.backward(&x, &y, 2, 8);
    let names = ["tok_emb", "pos_emb"];
    let block_names = [
        "ln1_g", "ln1_b", "w_qkv", "b_qkv", "w_o", "b_o", "ln2_g", "ln2_b", "w_1", "b_1", "w_2",
        "b_2",
    ];
    let tail = ["lnf_g", "lnf_b", "head", "head_b"];
    let mut labels: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    for l in 0..cfg.layers {
        labels.extend(block_names.iter().map(|s| format!("block{l}.{s}")));
    }
    labels.extend(tail.iter().map(|s| s.to_string()));
    println!(
        "Gradient check on a tiny model ({} parameters)\n",
        model.n_params()
    );
    println!("{:<16} {:>8} {:>14}", "parameter", "checked", "largest gap");
    // A very small nudge: bigger ones can push ReLU units across their on/off
    // point, which makes the direct measurement jump and look like an error.
    let h = 1e-6;
    let mut worst_all: f64 = 0.0;
    let grad_slices: Vec<Vec<f64>> = grads.slices_mut().iter().map(|s| s.to_vec()).collect();
    for (p, label) in labels.iter().enumerate() {
        let len = grad_slices[p].len();
        let mut worst: f64 = 0.0;
        let picks: Vec<usize> = (0..len.min(12)).map(|k| k * len / len.min(12)).collect();
        for &i in &picks {
            let mut plus = model.clone();
            plus.slices_mut()[p][i] += h;
            let mut minus = model.clone();
            minus.slices_mut()[p][i] -= h;
            let numeric = (plus.loss(&x, &y, 2, 8) - minus.loss(&x, &y, 2, 8)) / (2.0 * h);
            worst = worst.max((numeric - grad_slices[p][i]).abs());
        }
        worst_all = worst_all.max(worst);
        println!("{label:<16} {:>8} {worst:>14.1e}", picks.len());
    }
    println!("\nLargest gap anywhere: {worst_all:.1e}");
    Ok(())
}
pub fn train() -> AnyResult<()> {
    let (tok, train_ids, val_ids) = load()?;
    let cfg = Config {
        vocab: tok.chars.len(),
        context: 64,
        dim: 64,
        heads: 4,
        layers: 2,
    };
    let steps = 3000;
    let batch = 16;
    let mut model = Gpt::new(cfg, SEED);
    let mut adam = Adam::new(&model, 3e-3);
    let mut rng = StdRng::seed_from_u64(SEED);
    println!(
        "Tiny GPT: {} layers, {} heads, {}-number vectors, looks back {} characters, {} parameters",
        cfg.layers,
        cfg.heads,
        cfg.dim,
        cfg.context,
        model.n_params()
    );
    println!(
        "Training {steps} steps of {batch} × {} characters\n",
        cfg.context
    );
    println!(
        "{:>6} {:>11} {:>13} {:>9}",
        "step", "train loss", "held-back loss", "time"
    );
    let start = Instant::now();
    for step in 0..=steps {
        if step % 500 == 0 {
            println!(
                "{step:>6} {:>11.3} {:>13.3} {:>8.0}s",
                estimate_loss(&model, &train_ids, 8),
                estimate_loss(&model, &val_ids, 8),
                start.elapsed().as_secs_f64()
            );
        }
        if step == steps {
            break;
        }
        // Lower the step size for the last third of training.
        if step == steps * 2 / 3 {
            adam.lr = 1e-3;
        }
        let (x, y) = get_batch(&train_ids, batch, cfg.context, &mut rng);
        let (_, mut grads) = model.backward(&x, &y, batch, cfg.context);
        adam.step(&mut model, &mut grads);
    }
    println!("\nSample (temperature 0.8):\n");
    println!("{}", generate(&model, &tok, "ROMEO:\n", 400, 0.8, SEED));
    let saved = SavedGpt {
        tokenizer: tok,
        model,
    };
    std::fs::write(MODEL, serde_json::to_string(&saved)?)?;
    println!("\nSaved: {MODEL}");
    Ok(())
}
pub fn write() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "ROMEO:\n".to_string())
        .replace("\\n", "\n");
    let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(300);
    let temperature: f64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0.8);
    let saved: SavedGpt = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    println!(
        "{}",
        generate(
            &saved.model,
            &saved.tokenizer,
            &prompt,
            n,
            temperature,
            SEED
        )
    );
    Ok(())
}
/// Look inside: for the last character of a text, which earlier characters does each head attend to?
pub fn attend() -> AnyResult<()> {
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| {
            "KING HENRY:\nWhat says the
king?"
                .to_string()
        })
        .replace("\\n", "\n");
    let saved: SavedGpt = serde_json::from_str(&std::fs::read_to_string(MODEL)?)?;
    let (model, tok) = (&saved.model, &saved.tokenizer);
    let mut ids = tok.encode(&text);
    let start = ids.len().saturating_sub(model.config.context);
    ids = ids[start..].to_vec();
    let t = ids.len();
    let cache = model.forward(&ids, 1, t);
    // Show a line break as \n so it is visible.
    let show = |c: char| {
        if c == '\n' {
            "\\n".to_string()
        } else {
            c.to_string()
        }
    };
    println!("Text: {:?}", tok.decode(&ids));
    println!(
        "Predicting what comes after the final '{}'\n",
        show(tok.chars[ids[t - 1]])
    );
    for (l, bc) in cache.blocks.iter().enumerate() {
        for (h, p) in bc.probs.iter().enumerate() {
            let row = p.row(t - 1);
            let mut ranked: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top: Vec<String> = ranked
                .iter()
                .take(4)
                .map(|&(j, w)| {
                    format!(
                        "{:>3}% '{}' (pos {j})",
                        (w * 100.0).round(),
                        show(tok.chars[ids[j]])
                    )
                })
                .collect();
            println!(
                "layer {l} head {h}:
{}",
                top.join(
                    "
"
                )
            );
        }
    }
    let last = cache.probs.row(t - 1);
    let mut next: Vec<(usize, f64)> = last.iter().copied().enumerate().collect();
    next.sort_by(|a, b| b.1.total_cmp(&a.1));
    let guesses: Vec<String> = next
        .iter()
        .take(5)
        .map(|&(i, p)| format!("'{}' {:.0}%", show(tok.chars[i]), p * 100.0))
        .collect();
    println!("\nMost likely next characters: {}", guesses.join(", "));
    Ok(())
}
