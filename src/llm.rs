//! Chapter 11: a real open model, adapted with Candle.
//!
//! SmolLM2-135M (Hugging Face, Apache 2.0) is a Llama-style transformer with
//! 134.5 million weights, pretrained on trillions of tokens of web text, code
//! and maths. We write the model ourselves with Candle (Hugging Face's Rust
//! library), load its published weights, compare it with Chapter 10's tiny GPT,
//! then keep training it on Shakespeare.
//!
//! The big change from Chapter 10: Candle works out every gradient for us
//! (automatic differentiation), so there is no hand-written backward pass.
//!
//!tokens → embedding → [RMSNorm → attention (RoPE) → add back · RMSNorm → SwiGLU MLP → add back] × 30
//!→ RMSNorm → a score for each of the 49,152 tokens
use crate::AnyResult;
use crate::data::SEED;
use crate::gpt;
use candle_core::{D, DType, Device, IndexOp, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;
use tokenizers::Tokenizer;
/// The downloaded model: config.json, tokenizer.json, model.safetensors.
pub const DIR: &str = "models/smollm2-135m";
/// Our version after extra training on Shakespeare.
pub const TUNED: &str = "models/smollm2-shakespeare.safetensors";
/// Training changes only the layers above this one (the last 10 of 30) and the final norm.
const FROZEN_LAYERS: usize = 20;
// ================= shape =================
/// The fields we need from the model's config.json.
#[derive(Deserialize, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}
// ================= parts =================
/// RMSNorm: like LayerNorm, but only divides by the size of the vector (no average subtracted).
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(RmsNorm {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean_square = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(mean_square + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)
    }
}
/// A weight matrix with no bias: y = x · Wᵀ.
struct Linear {
    weight: Tensor, // outputs × inputs, as stored in the file
}
impl Linear {
    /// Flatten batch × tokens into one dimension so this is a single 2-D matrix multiply.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, n) = x.dims3()?;
        x.reshape((b * t, n))?
            .matmul(&self.weight.t()?)?
            .reshape((b, t, ()))
    }
}
fn linear(inputs: usize, outputs: usize, vb: VarBuilder) -> candle_core::Result<Linear> {
    Ok(Linear {
        weight: vb.get((outputs, inputs), "weight")?,
    })
}
/// Rotary position embedding (RoPE): instead of adding a position vector, rotate
/// pairs of numbers in each query and key by an angle that grows with position.
struct Rope {
    cos: Tensor, // positions × head_dim/2
    sin: Tensor,
}
impl Rope {
    fn new(cfg: &Config, device: &Device) -> candle_core::Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let positions = cfg.max_position_embeddings.min(4096);
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;
        let pos = Tensor::arange(0u32, positions as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((positions, 1))?;
        let angles = pos.matmul(&inv_freq)?;
        Ok(Rope {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }
    /// x: batch × heads × tokens × head_dim, starting at position `pos`.
    fn apply(&self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        let (_, _, t, hd) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, t)?;
        let sin = self.sin.narrow(0, pos, t)?;
        let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
        let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)
    }
}
/// Saved keys and values for each layer, so writing a new token does not
/// recompute the whole text (Chapter 10's "KV cache" improvement).
pub type KvCache = Vec<Option<(Tensor, Tensor)>>;
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}
impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_size = head_dim * cfg.num_key_value_heads;
        Ok(Attention {
            q: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k: linear(cfg.hidden_size, kv_size, vb.pp("k_proj"))?,
            v: linear(cfg.hidden_size, kv_size, vb.pp("v_proj"))?,
            o: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        rope: &Rope,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
    ) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |y: Tensor, n: usize| {
            y.reshape((b, t, n, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = rope.apply(&split(self.q.forward(x)?, self.heads)?, pos)?;
        let mut k = rope.apply(&split(self.k.forward(x)?, self.kv_heads)?, pos)?;
        let mut v = split(self.v.forward(x)?, self.kv_heads)?;
        if let Some(slot) = cache {
            if let Some((old_k, old_v)) = slot.as_ref() {
                k = Tensor::cat(&[old_k, &k], 2)?;
                v = Tensor::cat(&[old_v, &v], 2)?;
            }
            *slot = Some((k.clone(), v.clone()));
        }
        // Grouped-query attention: 9 query heads share 3 key/value heads, 3 each.
        let rep = self.heads / self.kv_heads;
        let k = repeat_kv(k, rep)?;
        let v = repeat_kv(v, rep)?;
        let mut scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        if t > 1 {
            scores = scores.broadcast_add(&causal_mask(t, pos, x.device())?)?;
        }
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let y = weights
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o.forward(&y)
    }
}
/// Copy each key/value head so it lines up with the query heads that share it.
fn repeat_kv(x: Tensor, rep: usize) -> candle_core::Result<Tensor> {
    if rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, t, hd) = x.dims4()?;
    Tensor::cat(&vec![&x; rep], 2)?.reshape((b, kv_heads * rep, t, hd))
}
/// 0 where a token may look, minus infinity where it would be looking ahead.
fn causal_mask(t: usize, pos: usize, device: &Device) -> candle_core::Result<Tensor> {
    let total = pos + t;
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| (0..total).map(move |j| if j > pos + i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (t, total), device)
}
/// SwiGLU MLP: one path decides how much of the other to let through.
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Mlp {
            gate: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&h)
    }
}
struct Layer {
    norm1: RmsNorm,
    attn: Attention,
    norm2: RmsNorm,
    mlp: Mlp,
}
// ================= the model =================
pub struct Llm {
    pub cfg: Config,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    head: Tensor,
    rope: Rope,
}
impl Llm {
    fn new(cfg: Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed = vb.get(
            (cfg.vocab_size, cfg.hidden_size),
            "model.embed_tokens.weight",
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb.pp(format!("model.layers.{i}"));
                Ok(Layer {
                    norm1: RmsNorm::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("input_layernorm"),
                    )?,
                    attn: Attention::new(&cfg, vb.pp("self_attn"))?,
                    norm2: RmsNorm::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_attention_layernorm"),
                    )?,
                    mlp: Mlp::new(&cfg, vb.pp("mlp"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        // SmolLM2 reuses the embedding table to score the next token ("tied" weights).
        let head = if cfg.tie_word_embeddings {
            embed.clone()
        } else {
            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        };
        let rope = Rope::new(&cfg, vb.device())?;
        Ok(Llm {
            cfg,
            embed,
            layers,
            norm,
            head,
            rope,
        })
    }
    /// Token ids (batch × tokens) → the final vectors (batch × tokens × hidden).
    pub fn hidden(
        &self,
        ids: &Tensor,
        pos: usize,
        mut cache: Option<&mut KvCache>,
    ) -> candle_core::Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let mut x = self.embed.index_select(&ids.flatten_all()?, 0)?.reshape((
            b,
            t,
            self.cfg.hidden_size,
        ))?;
        for (i, layer) in self.layers.iter().enumerate() {
            let slot = cache.as_deref_mut().map(|c| &mut c[i]);
            let a = layer
                .attn
                .forward(&layer.norm1.forward(&x)?, pos, &self.rope, slot)?;
            x = (x + a)?;
            let m = layer.mlp.forward(&layer.norm2.forward(&x)?)?;
            x = (x + m)?;
        }
        self.norm.forward(&x)
    }
    /// Final vectors → a score for every token in the vocabulary.
    pub fn logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, h) = hidden.dims3()?;
        // One flat matrix multiply against the transposed table: no copy of its 28 million numbers.
        hidden
            .reshape((b * t, h))?
            .matmul(&self.head.t()?)?
            .reshape((b, t, self.cfg.vocab_size))
    }
    pub fn new_cache(&self) -> KvCache {
        vec![None; self.cfg.num_hidden_layers]
    }
}
// ================= speed =================
/// Treat numbers too small to matter (below about 1e-38, "denormals") as zero.
/// Otherwise x86 processors slow down up to 50 times on them, and gradients
/// shrinking through 30 layers produce plenty. Deep-learning libraries do the same.
fn flush_denormals() {
    #[cfg(target_arch = "x86_64")]
    fn set() {
        // Bits 15 and 6 of the MXCSR register: flush-to-zero and denormals-are-zero.
        #[allow(deprecated)]
        unsafe {
            use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
            _mm_setcsr(_mm_getcsr() | 0x8040);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    fn set() {}
    set();
    // Candle's matrix multiplies run on rayon's worker threads; switch it on in each of them too.
    let _ = rayon::ThreadPoolBuilder::new()
        .start_handler(|_| set())
        .build_global();
}
// ================= loading and saving =================
pub struct Loaded {
    pub model: Llm,
    /// Every weight as a 32-bit tensor, by name. Trainable ones share storage with `trainable`.
    pub tensors: HashMap<String, Tensor>,
    pub trainable: Vec<Var>,
    pub tokenizer: Tokenizer,
}
/// Read config and tokenizer from DIR and the weights from `weights`.
/// Weights whose names pass `train` become variables that Candle tracks for gradients;
/// the rest are plain tensors, so no memory or time is spent on their gradients.
pub fn load(weights: &str, train: &dyn Fn(&str) -> bool) -> AnyResult<Loaded> {
    flush_denormals();
    let device = Device::Cpu;
    let cfg: Config =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/config.json"))?)?;
    let tokenizer =
        Tokenizer::from_file(format!("{DIR}/tokenizer.json")).map_err(|e| e.to_string())?;
    let mut tensors = HashMap::new();
    let mut trainable = Vec::new();
    for (name, t) in candle_core::safetensors::load(weights, &device)? {
        let t = t.to_dtype(DType::F32)?;
        if train(&name) {
            let var = Var::from_tensor(&t)?;
            tensors.insert(name, var.as_tensor().clone());
            trainable.push(var);
        } else {
            tensors.insert(name, t);
        }
    }
    // A misspelt name is an error here, never a silently random weight.
    let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
    let model = Llm::new(cfg, vb)?;
    Ok(Loaded {
        model,
        tensors,
        trainable,
        tokenizer,
    })
}
/// Nothing trainable: for writing and measuring.
pub fn load_frozen(weights: &str) -> AnyResult<Loaded> {
    load(weights, &|_| false)
}
/// The last 10 layers and the final norm.
fn top_layers(name: &str) -> bool {
    let layer = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|n| n.parse::<usize>().ok());
    name == "model.norm.weight" || layer.is_some_and(|n| n >= FROZEN_LAYERS)
}
/// Save every weight as 16-bit (bfloat16), like the original file: half the size.
fn save(tensors: &HashMap<String, Tensor>, path: &str) -> AnyResult<()> {
    let mut out = HashMap::new();
    for (name, t) in tensors {
        out.insert(name.clone(), t.to_dtype(DType::BF16)?);
    }
    candle_core::safetensors::save(&out, path)?;
    Ok(())
}
pub fn n_params(tensors: &HashMap<String, Tensor>) -> usize {
    tensors.values().map(|t| t.elem_count()).sum()
}
fn encode(tok: &Tokenizer, text: &str) -> AnyResult<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}
fn decode(tok: &Tokenizer, ids: &[u32]) -> AnyResult<String> {
    Ok(tok.decode(ids, false).map_err(|e| e.to_string())?)
}
/// Shakespeare split exactly as in Chapter 10: first 90% of characters, last 10% held back.
fn shakespeare() -> AnyResult<(String, String)> {
    let text = std::fs::read_to_string(gpt::TEXT)?;
    let chars: Vec<char> = text.chars().collect();
    let cut = chars.len() * 9 / 10;
    Ok((chars[..cut].iter().collect(), chars[cut..].iter().collect()))
}
// ================= writing =================
/// Write up to `n` new tokens after `prompt`, keeping keys and values in a cache.
pub fn generate(
    loaded: &Loaded,
    prompt: &str,
    n: usize,
    temperature: f64,
    seed: u64,
) -> AnyResult<String> {
    let model = &loaded.model;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids = encode(&loaded.tokenizer, prompt)?;
    if ids.is_empty() {
        ids = encode(&loaded.tokenizer, "\n")?;
    }
    let mut cache = model.new_cache();
    let mut input = Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let mut pos = 0;
    for _ in 0..n {
        let t = input.dim(1)?;
        let h = model.hidden(&input, pos, Some(&mut cache))?;
        let last = model.logits(&h.i((.., t - 1..t, ..))?)?.flatten_all()?;
        pos += t;
        let next = sample(&last, temperature, &mut rng)?;
        if next == 0 {
            break; // <|endoftext|>
        }
        ids.push(next);
        input = Tensor::new(&[next], &Device::Cpu)?.unsqueeze(0)?;
        if pos + 1 >= model.rope.cos.dim(0)? {
            break;
        }
    }
    decode(&loaded.tokenizer, &ids)
}
/// Pick a token at random, weighted by its probability after temperature.
fn sample(logits: &Tensor, temperature: f64, rng: &mut StdRng) -> AnyResult<u32> {
    let probs = candle_nn::ops::softmax_last_dim(&(logits / temperature)?)?.to_vec1::<f32>()?;
    let mut r = rng.random::<f32>();
    for (i, p) in probs.iter().enumerate() {
        if r < *p {
            return Ok(i as u32);
        }
        r -= p;
    }
    Ok(probs.len() as u32 - 1)
}
// ================= measuring =================
/// Total loss (in nats) of predicting every token of `ids` from the ones before it,
/// reading the text in windows of `window` tokens.
fn total_loss(model: &Llm, ids: &[u32], window: usize) -> AnyResult<f64> {
    let mut total = 0.0;
    for s in (0..ids.len().saturating_sub(1)).step_by(window) {
        let end = (s + window).min(ids.len() - 1);
        let x = Tensor::new(&ids[s..end], &Device::Cpu)?.unsqueeze(0)?;
        let y = Tensor::new(&ids[s + 1..end + 1], &Device::Cpu)?;
        let logits = model.logits(&model.hidden(&x, 0, None)?)?.squeeze(0)?;
        total += candle_nn::loss::cross_entropy(&logits, &y)?.to_scalar::<f32>()? as f64
            * (end - s) as f64;
    }
    Ok(total)
}
/// Chapter 10's tiny GPT on the same text, in windows of its 64 characters.
fn tiny_gpt_loss_per_char(text: &str) -> AnyResult<f64> {
    let saved: gpt::SavedGpt = serde_json::from_str(&std::fs::read_to_string(gpt::MODEL)?)?;
    let ids = saved.tokenizer.encode(text);
    let t = saved.model.config.context;
    let mut total = 0.0;
    let mut count = 0;
    let mut s = 0;
    while s + t < ids.len() {
        total += saved
            .model
            .loss(&ids[s..s + t], &ids[s + 1..s + t + 1], 1, t)
            * t as f64;
        count += t;
        s += t;
    }
    Ok(total / count as f64)
}
// ================= stages =================
/// Step 1: load the model, look at its tokens, and check our code against Candle's own Llama.
pub fn check() -> AnyResult<()> {
    let start = Instant::now();
    let loaded = load_frozen(&format!("{DIR}/model.safetensors"))?;
    let cfg = &loaded.model.cfg;
    println!("Loaded in {:.1}s", start.elapsed().as_secs_f64());
    println!(
"SmolLM2-135M: {} layers, {} query heads sharing {} key/value heads, {}-number vectors, {} tokens in
the vocabulary",
cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size,
cfg.vocab_size
);
    println!("Weights: {}\n", n_params(&loaded.tensors));
    let text = "ROMEO:\nWhat light through yonder window breaks?";
    let ids = encode(&loaded.tokenizer, text)?;
    let pieces: Vec<String> = ids
        .iter()
        .map(|&id| format!("{:?}", decode(&loaded.tokenizer, &[id]).unwrap_or_default()))
        .collect();
    println!(
        "{} characters become {} tokens:",
        text.chars().count(),
        ids.len()
    );
    println!("{}\n", pieces.join(" "));
    // Check 1: our model against Candle's ready-made Llama, same weights, same text.
    let x = Tensor::new(ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let t = ids.len();
    let ours = loaded
        .model
        .logits(&loaded.model.hidden(&x, 0, None)?)?
        .i((0, t - 1))?;
    let their_cfg: candle_transformers::models::llama::LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/config.json"))?)?;
    let their_cfg = their_cfg.into_config(false);
    let vb = VarBuilder::from_tensors(loaded.tensors.clone(), DType::F32, &Device::Cpu);
    let theirs_model = candle_transformers::models::llama::Llama::load(vb, &their_cfg)?;
    let mut their_cache = candle_transformers::models::llama::Cache::new(
        false,
        DType::F32,
        &their_cfg,
        &Device::Cpu,
    )?;
    let theirs = theirs_model.forward(&x, 0, &mut their_cache)?.squeeze(0)?;
    let biggest = ours.abs()?.max(0)?.to_scalar::<f32>()?;
    let gap1 = (ours.clone() - theirs)?.abs()?.max(0)?.to_scalar::<f32>()?;
    println!("Scores for the next token range up to {biggest:.1}");
    println!("Largest gap between our model and Candle's Llama: {gap1:.1e}");
    // Check 2: writing with the key/value cache must give the same scores as recomputing everything.
    let mut cache = loaded.model.new_cache();
    let first = x.narrow(1, 0, t - 1)?;
    loaded.model.hidden(&first, 0, Some(&mut cache))?;
    let last = x.narrow(1, t - 1, 1)?;
    let cached = loaded
        .model
        .logits(&loaded.model.hidden(&last, t - 1, Some(&mut cache))?)?
        .i((0, 0))?;
    let gap2 = (ours.clone() - cached)?.abs()?.max(0)?.to_scalar::<f32>()?;
    println!("Largest gap between cached and full recompute: {gap2:.1e}");
    let probs = candle_nn::ops::softmax_last_dim(&ours)?.to_vec1::<f32>()?;
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let top: Vec<String> = order[..5]
        .iter()
        .map(|&i| {
            format!(
                "{:?} {:.0}%",
                decode(&loaded.tokenizer, &[i as u32]).unwrap_or_default(),
                probs[i] * 100.0
            )
        })
        .collect();
    println!("\nMost likely next tokens: {}", top.join(", "));
    Ok(())
}
/// Step 2: write text. Arguments: prompt, tokens, temperature, and "tuned" to use our trained version.
pub fn write() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "ROMEO:\n".to_string())
        .replace("\\n", "\n");
    let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(100);
    let temperature: f64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0.8);
    let tuned = args.get(5).is_some_and(|s| s == "tuned");
    let weights = if tuned {
        TUNED.to_string()
    } else {
        format!("{DIR}/model.safetensors")
    };
    let loaded = load_frozen(&weights)?;
    let start = Instant::now();
    let text = generate(&loaded, &prompt, n, temperature, SEED)?;
    let secs = start.elapsed().as_secs_f64();
    println!("{text}");
    let written = encode(&loaded.tokenizer, &text)?
        .len()
        .saturating_sub(encode(&loaded.tokenizer, &prompt)?.len());
    println!(
        "\n[{} model, {written} tokens in {secs:.1}s, {:.0} tokens a second]",
        if tuned { "tuned" } else { "original" },
        written as f64 / secs
    );
    Ok(())
}
/// Step 3: both models on the same held-back Shakespeare, measured per character.
pub fn score() -> AnyResult<()> {
    let (_, held_back) = shakespeare()?;
    let chars = held_back.chars().count();
    println!("Held-back Shakespeare: {chars} characters (the last 10%, as in Chapter 10)\n");
    let tiny = tiny_gpt_loss_per_char(&held_back)?;
    println!("{:<28} {:>10} {:>14}", "Model", "tokens", "loss per char");
    println!(
        "{:<28} {:>10} {:>14.3}",
        "Tiny GPT (Chapter 10)", chars, tiny
    );
    let mut runs = vec![("SmolLM2-135M, original", format!("{DIR}/model.safetensors"))];
    if std::path::Path::new(TUNED).exists() {
        runs.push(("SmolLM2-135M, tuned", TUNED.to_string()));
    }
    for (name, weights) in runs {
        let loaded = load_frozen(&weights)?;
        let ids = encode(&loaded.tokenizer, &held_back)?;
        let start = Instant::now();
        let total = total_loss(&loaded.model, &ids, 256)?;
        println!(
            "{:<28} {:>10} {:>14.3}
({:.0}s)",
            name,
            ids.len(),
            total / chars as f64,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
/// Step 4: keep training the pretrained model on the Shakespeare training text.
pub fn train() -> AnyResult<()> {
    let (train_text, held_back) = shakespeare()?;
    let loaded = load(&format!("{DIR}/model.safetensors"), &top_layers)?;
    let train_ids = encode(&loaded.tokenizer, &train_text)?;
    let held_ids = encode(&loaded.tokenizer, &held_back)?;
    // A fixed slice of held-back text to watch during training (the full score is Step 3).
    let watch = &held_ids[..held_ids.len().min(2048)];
    let watch_chars = decode(&loaded.tokenizer, watch)?.chars().count() as f64;
    let steps = 200;
    let (batch, window) = (4, 128);
    let lr = 1e-4;
    let mut opt = AdamW::new(
        loaded.trainable.clone(),
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let mut rng = StdRng::seed_from_u64(SEED);
    let n_train: usize = loaded.trainable.iter().map(|v| v.elem_count()).sum();
    println!(
        "Training the last {} of {} layers: {} of {} weights (the rest stay frozen)",
        loaded.model.cfg.num_hidden_layers - FROZEN_LAYERS,
        loaded.model.cfg.num_hidden_layers,
        n_train,
        n_params(&loaded.tensors)
    );
    println!(
        "Training tokens: {}, held-back tokens: {}",
        train_ids.len(),
        held_ids.len()
    );
    println!(
        "{steps} steps of {batch} × {window} tokens, step size {lr}, warming up over the first 20
steps\n"
    );
    println!(
        "{:>6} {:>12} {:>16} {:>15} {:>8}",
        "step", "train/token", "held-back/token", "held-back/char", "time"
    );
    let start = Instant::now();
    let mut recent = Vec::new();
    for step in 0..=steps {
        if step % 25 == 0 {
            let held = total_loss(&loaded.model, watch, 256)?;
            let train_loss = if recent.is_empty() {
                "-".to_string()
            } else {
                format!("{:.3}", recent.iter().sum::<f64>() / recent.len() as f64)
            };
            println!(
                "{step:>6} {train_loss:>12} {:>16.3} {:>15.3} {:>7.0}s",
                held / watch.len() as f64,
                held / watch_chars,
                start.elapsed().as_secs_f64()
            );
            recent.clear();
        }
        if step == steps {
            break;
        }
        // Warm up: start with tiny steps so the first updates do not damage what the model knows.
        opt.set_learning_rate(lr * ((step + 1) as f64 / 20.0).min(1.0));
        let mut xs = Vec::with_capacity(batch * window);
        let mut ys = Vec::with_capacity(batch * window);
        for _ in 0..batch {
            let s = rng.random_range(0..train_ids.len() - window - 1);
            xs.extend_from_slice(&train_ids[s..s + window]);
            ys.extend_from_slice(&train_ids[s + 1..s + window + 1]);
        }
        let x = Tensor::from_vec(xs, (batch, window), &Device::Cpu)?;
        let y = Tensor::from_vec(ys, batch * window, &Device::Cpu)?;
        let logits = loaded.model.logits(&loaded.model.hidden(&x, 0, None)?)?;
        let loss = candle_nn::loss::cross_entropy(
            &logits.reshape((batch * window, loaded.model.cfg.vocab_size))?,
            &y,
        )?;
        opt.backward_step(&loss)?; // Candle works out every gradient, then AdamW updates every weight
        recent.push(loss.to_scalar::<f32>()? as f64);
    }
    save(&loaded.tensors, TUNED)?;
    println!("\nSaved: {TUNED}");
    Ok(())
}
