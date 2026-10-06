//! Chapter 9: a neural network, written from scratch.
//!
//! Layers of weights with ReLU in between, softmax at the end,
//! backpropagation to work out how every weight should change,
//! and the Adam optimizer to make the changes.
use crate::AnyResult;
use crate::data::{self, FOLDS, SEED};
use crate::metrics::{auc, best_threshold, calibration, report};
use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
// ---------- the network ----------
/// One fully connected layer: output = input · weights + bias.
#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    pub w: Array2<f64>, // inputs × outputs
    pub b: Array1<f64>, // outputs
}
#[derive(Serialize, Deserialize, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}
/// A normally distributed random number (Box–Muller), no extra crate needed.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
impl Network {
    /// `sizes` like [784, 128, 10]: inputs, hidden layers, classes.
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let layers = sizes
            .windows(2)
            .map(|pair| {
                let (n_in, n_out) = (pair[0], pair[1]);
                // He initialisation: random weights sized so signals neither
                // explode nor fade as they pass through ReLU layers.
                let scale = (2.0 / n_in as f64).sqrt();
                Layer {
                    w: Array2::from_shape_fn((n_in, n_out), |_| gaussian(&mut rng) * scale),
                    b: Array1::zeros(n_out),
                }
            })
            .collect();
        Network { layers }
    }
    /// Forward pass. Returns every layer's input (needed by backprop) and the class probabilities.
    fn forward_all(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Array2<f64>) {
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut a = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            inputs.push(a.clone());
            let z = a.dot(&layer.w) + &layer.b;
            a = if i + 1 < self.layers.len() {
                z.mapv(|v| v.max(0.0)) // ReLU: keep positives, zero the rest
            } else {
                softmax(&z)
            };
        }
        (inputs, a)
    }
    /// Class probabilities for each row of `x`.
    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        self.forward_all(x).1
    }
    /// Average cross-entropy: how surprised the network is by the right answers.
    pub fn loss(&self, x: &Array2<f64>, y: &[usize]) -> f64 {
        let p = self.predict(x);
        -y.iter()
            .enumerate()
            .map(|(i, &c)| p[[i, c]].max(1e-12).ln())
            .sum::<f64>()
            / y.len() as f64
    }
    /// Backpropagation: the gradient of the loss for every weight and bias.
    pub fn gradients(&self, x: &Array2<f64>, y: &[usize], weight_decay: f64) -> Vec<Layer> {
        let n = x.nrows() as f64;
        let (inputs, probs) = self.forward_all(x);
        // At the output, softmax + cross-entropy has a simple gradient:
        // (predicted probability − 1 for the right class, or − 0 for the others) / n
        let mut dz = probs;
        for (i, &c) in y.iter().enumerate() {
            dz[[i, c]] -= 1.0;
        }
        dz /= n;
        let mut grads = Vec::with_capacity(self.layers.len());
        for (l, layer) in self.layers.iter().enumerate().rev() {
            let a_prev = &inputs[l];
            let mut dw = a_prev.t().dot(&dz);
            if weight_decay > 0.0 {
                dw = dw + &layer.w * weight_decay;
            }
            let db = dz.sum_axis(Axis(0));
            if l > 0 {
                // Pass the blame back one layer, through the ReLU:
                // units that were switched off (input ≤ 0) get none.
                let da = dz.dot(&layer.w.t());
                dz = da * &a_prev.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            }
            grads.push(Layer { w: dw, b: db });
        }
        grads.reverse();
        grads
    }
}
fn softmax(z: &Array2<f64>) -> Array2<f64> {
    let mut out = z.clone();
    for mut row in out.rows_mut() {
        let max = row.fold(f64::MIN, |m, &v| m.max(v)); // subtract the max so exp() never overflows
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        row /= sum;
    }
    out
}
// ---------- training: Adam ----------
/// Adam keeps a running average of each weight's gradient (direction)
/// and of its square (typical size), and steps by their ratio.
struct Adam {
    lr: f64,
    t: i32,
    m: Vec<Layer>,
    v: Vec<Layer>,
}
impl Adam {
    fn new(net: &Network, lr: f64) -> Self {
        let zeros: Vec<Layer> = net
            .layers
            .iter()
            .map(|l| Layer {
                w: Array2::zeros(l.w.raw_dim()),
                b: Array1::zeros(l.b.len()),
            })
            .collect();
        Adam {
            lr,
            t: 0,
            m: zeros.clone(),
            v: zeros,
        }
    }
    fn step(&mut self, net: &mut Network, grads: &[Layer]) {
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        self.t += 1;
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        for ((layer, g), (m, v)) in net
            .layers
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            m.w.zip_mut_with(&g.w, |m, &g| *m = b1 * *m + (1.0 - b1) * g);
            v.w.zip_mut_with(&g.w, |v, &g| *v = b2 * *v + (1.0 - b2) * g * g);
            m.b.zip_mut_with(&g.b, |m, &g| *m = b1 * *m + (1.0 - b1) * g);
            v.b.zip_mut_with(&g.b, |v, &g| *v = b2 * *v + (1.0 - b2) * g * g);
            let lr = self.lr;
            ndarray::Zip::from(&mut layer.w)
                .and(&m.w)
                .and(&v.w)
                .for_each(|w, &m, &v| {
                    *w -= lr * (m / c1) / ((v / c2).sqrt() + eps);
                });
            ndarray::Zip::from(&mut layer.b)
                .and(&m.b)
                .and(&v.b)
                .for_each(|b, &m, &v| {
                    *b -= lr * (m / c1) / ((v / c2).sqrt() + eps);
                });
        }
    }
}
#[derive(Clone, Copy)]
pub struct TrainSettings {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
}
/// Mini-batch training. `report` is called after every epoch with (epoch, network).
pub fn train(
    net: &mut Network,
    x: &Array2<f64>,
    y: &[usize],
    settings: TrainSettings,
    mut report: impl FnMut(usize, &Network),
) {
    let mut adam = Adam::new(net, settings.lr);
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut order: Vec<usize> = (0..x.nrows()).collect();
    for epoch in 1..=settings.epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(settings.batch) {
            let xb = x.select(Axis(0), chunk);
            let yb: Vec<usize> = chunk.iter().map(|&i| y[i]).collect();
            let grads = net.gradients(&xb, &yb, settings.weight_decay);
            adam.step(net, &grads);
        }
        report(epoch, net);
    }
}
pub fn accuracy(net: &Network, x: &Array2<f64>, y: &[usize]) -> f64 {
    let p = net.predict(x);
    let correct = p
        .rows()
        .into_iter()
        .zip(y)
        .filter(|(row, c)| crate::svm::argmax(&row.to_vec()) == **c)
        .count();
    correct as f64 / y.len() as f64
}
// ---------- stage: XOR and a gradient check ----------
/// Compare backprop's gradients with slow, simple finite differences.
fn gradient_check(net: &Network, x: &Array2<f64>, y: &[usize]) -> f64 {
    let grads = net.gradients(x, y, 0.0);
    let h = 1e-5;
    let mut worst: f64 = 0.0;
    for (l, layer) in net.layers.iter().enumerate() {
        for idx in 0..layer.w.len() {
            let (r, c) = (idx / layer.w.ncols(), idx % layer.w.ncols());
            let mut plus = net.clone();
            plus.layers[l].w[[r, c]] += h;
            let mut minus = net.clone();
            minus.layers[l].w[[r, c]] -= h;
            let numeric = (plus.loss(x, y) - minus.loss(x, y)) / (2.0 * h);
            worst = worst.max((numeric - grads[l].w[[r, c]]).abs());
        }
    }
    worst
}
pub fn xor() -> AnyResult<()> {
    // The four XOR cases: output is 1 when exactly one input is 1.
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = vec![0, 1, 1, 0];
    let settings = TrainSettings {
        epochs: 500,
        batch: 4,
        lr: 0.05,
        weight_decay: 0.0,
    };
    // 1. No hidden layer: a straight line cannot separate XOR.
    let mut linear = Network::new(&[2, 2], SEED);
    train(&mut linear, &x, &y, settings, |_, _| {});
    println!(
        "No hidden layer:
accuracy {:.0}%
loss {:.3}",
        accuracy(&linear, &x, &y) * 100.0,
        linear.loss(&x, &y)
    );
    // 2. One hidden layer of 8 ReLU units.
    let mut net = Network::new(&[2, 8, 2], SEED);
    println!(
        "\nGradient check (largest gap between backprop and finite differences): {:.1e}",
        gradient_check(&net, &x, &y)
    );
    println!("\nOne hidden layer, training:");
    train(&mut net, &x, &y, settings, |epoch, n| {
        if [1, 10, 50, 100, 200, 500].contains(&epoch) {
            println!(
                "
epoch {epoch:>3}: loss {:.4}",
                n.loss(&x, &y)
            );
        }
    });
    println!("\nPredictions:");
    let p = net.predict(&x);
    for i in 0..4 {
        println!(
            "
{} XOR {} = {}
(network: {:.3})",
            x[[i, 0]],
            x[[i, 1]],
            y[i],
            p[[i, 1]]
        );
    }
    println!("Accuracy: {:.0}%", accuracy(&net, &x, &y) * 100.0);
    Ok(())
}
// ---------- stage: the heart data ----------
pub const HEART_MODEL: &str = "nn_heart.json";
/// A trained heart network plus the scaling it expects.
#[derive(Serialize, Deserialize)]
pub struct SavedHeartNet {
    pub features: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub threshold: f64,
    pub net: Network,
}
impl SavedHeartNet {
    pub fn probs(&self, x: &Array2<f64>) -> Vec<f64> {
        let scaled = scale(x, &self.means, &self.stds);
        self.net.predict(&scaled).column(1).to_vec()
    }
}
fn column_stats(x: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let means = x.columns().into_iter().map(|c| c.mean().unwrap()).collect();
    let stds = x.columns().into_iter().map(|c| c.std(0.0)).collect();
    (means, stds)
}
fn scale(x: &Array2<f64>, means: &[f64], stds: &[f64]) -> Array2<f64> {
    let mut out = x.clone();
    for (j, mut col) in out.columns_mut().into_iter().enumerate() {
        col.mapv_inplace(|v| (v - means[j]) / stds[j]);
    }
    out
}
pub fn heart() -> AnyResult<()> {
    let df = data::load_csv(data::TRAIN)?;
    let features = data::feature_names(&df);
    let (x, y) = data::to_arrays(&df, &features)?;
    let labels: Vec<usize> = y.iter().map(|&b| b as usize).collect();
    let actual: Vec<bool> = y.to_vec();
    let fold_of = data::assign_folds(&y);
    let hiddens = [4, 16, 64];
    let decays = [0.0, 0.01, 0.1];
    let base = TrainSettings {
        epochs: 40,
        batch: 64,
        lr: 0.001,
        weight_decay: 0.0,
    };
    println!("Neural network 14 → hidden → 2, 5-fold cross-validation\n");
    println!("{:>7} {:>13} {:>7}", "hidden", "weight decay", "AUC");
    let mut best: Option<(usize, f64, f64, Vec<f64>)> = None;
    for &hidden in &hiddens {
        for &weight_decay in &decays {
            let settings = TrainSettings {
                weight_decay,
                ..base
            };
            let mut probs = vec![0.0; labels.len()];
            for fold in 0..FOLDS {
                let fit: Vec<usize> = (0..labels.len()).filter(|&i| fold_of[i] != fold).collect();
                let check: Vec<usize> = (0..labels.len()).filter(|&i| fold_of[i] == fold).collect();
                let x_fit = x.select(Axis(0), &fit);
                let (means, stds) = column_stats(&x_fit);
                let y_fit: Vec<usize> = fit.iter().map(|&i| labels[i]).collect();
                let mut net = Network::new(&[features.len(), hidden, 2], SEED);
                train(
                    &mut net,
                    &scale(&x_fit, &means, &stds),
                    &y_fit,
                    settings,
                    |_, _| {},
                );
                let p = net.predict(&scale(&x.select(Axis(0), &check), &means, &stds));
                for (k, &row) in check.iter().enumerate() {
                    probs[row] = p[[k, 1]];
                }
            }
            let a = auc(&probs, &actual);
            println!("{hidden:>7} {weight_decay:>13} {a:>7.3}");
            if best.as_ref().is_none_or(|b| a > b.2) {
                best = Some((hidden, weight_decay, a, probs));
            }
        }
    }
    let (hidden, weight_decay, _, probs) = best.expect("grid is not empty");
    let (threshold, f1) = best_threshold(&probs, &actual);
    println!(
        "\nBest: {hidden} hidden units, weight decay {weight_decay}, cut-off {threshold:.2} (F1
{f1:.3})"
    );
    println!("\nCalibration (training rows, out-of-fold):");
    calibration(&probs, &actual);
    // Final network on all training rows, tested once.
    let (means, stds) = column_stats(&x);
    let mut net = Network::new(&[features.len(), hidden, 2], SEED);
    train(
        &mut net,
        &scale(&x, &means, &stds),
        &labels,
        TrainSettings {
            weight_decay,
            ..base
        },
        |_, _| {},
    );
    let model = SavedHeartNet {
        features,
        means,
        stds,
        threshold,
        net,
    };
    let test_df = data::load_csv(data::TEST)?;
    let (x_test, y_test) = data::to_arrays(&test_df, &model.features)?;
    let test_probs = model.probs(&x_test);
    println!("\nTest AUC: {:.3}", auc(&test_probs, &y_test.to_vec()));
    report(threshold, &test_probs, &y_test.to_vec());
    std::fs::write(HEART_MODEL, serde_json::to_string(&model)?)?;
    println!("\nSaved: {HEART_MODEL}");
    Ok(())
}
// ---------- stage: handwritten digits ----------
pub const DIGITS_MODEL: &str = "nn_digits.json";
/// Read an MNIST image file: a 16-byte header, then 28×28 bytes per image.
fn load_images(path: &str) -> AnyResult<Array2<f64>> {
    let bytes = std::fs::read(path)?;
    let count = u32::from_be_bytes(bytes[4..8].try_into()?) as usize;
    let pixels: Vec<f64> = bytes[16..].iter().map(|&b| b as f64 / 255.0).collect();
    Ok(Array2::from_shape_vec((count, 784), pixels)?)
}
/// Read an MNIST label file: an 8-byte header, then one byte per label.
fn load_labels(path: &str) -> AnyResult<Vec<usize>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes[8..].iter().map(|&b| b as usize).collect())
}
pub fn load_mnist() -> AnyResult<(Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>)> {
    Ok((
        load_images("data/mnist/train-images-idx3-ubyte")?,
        load_labels("data/mnist/train-labels-idx1-ubyte")?,
        load_images("data/mnist/t10k-images-idx3-ubyte")?,
        load_labels("data/mnist/t10k-labels-idx1-ubyte")?,
    ))
}
/// Draw a 28×28 image with characters, darker pixels as denser symbols.
pub fn ascii(pixels: ndarray::ArrayView1<f64>) -> String {
    let shades = [' ', '.', ':', '+', '#', '@'];
    let mut out = String::new();
    for r in 0..28 {
        for c in 0..28 {
            let v = pixels[r * 28 + c];
            out.push(shades[((v * 5.0).round() as usize).min(5)]);
        }
        out.push('\n');
    }
    out
}
pub fn digits() -> AnyResult<()> {
    let (x_all, y_all, x_test, y_test) = load_mnist()?;
    // Hold back the last 10,000 training images to watch progress; the test set stays untouched.
    let (x_train, x_val) = (
        x_all.slice(ndarray::s![..50_000, ..]).to_owned(),
        x_all.slice(ndarray::s![50_000.., ..]).to_owned(),
    );
    let (y_train, y_val) = (y_all[..50_000].to_vec(), y_all[50_000..].to_vec());
    println!(
        "Training on {} images, watching {} held-back images\n",
        x_train.nrows(),
        x_val.nrows()
    );
    let settings = TrainSettings {
        epochs: 10,
        batch: 64,
        lr: 0.001,
        weight_decay: 0.0,
    };
    println!("No hidden layer (784 → 10):");
    let mut linear = Network::new(&[784, 10], SEED);
    train(&mut linear, &x_train, &y_train, settings, |epoch, n| {
        if epoch == settings.epochs {
            println!(
                "
after {epoch} epochs: held-back accuracy {:.2}%",
                accuracy(n, &x_val, &y_val) * 100.0
            );
        }
    });
    println!("\nOne hidden layer (784 → 128 → 10):");
    let mut net = Network::new(&[784, 128, 10], SEED);
    let start = std::time::Instant::now();
    train(&mut net, &x_train, &y_train, settings, |epoch, n| {
        println!(
            "
epoch {epoch:>2}: training loss {:.4}
held-back accuracy {:.2}%",
            n.loss(&x_train, &y_train),
            accuracy(n, &x_val, &y_val) * 100.0
        );
    });
    println!(
        "
training time: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    // The test set, once.
    let probs = net.predict(&x_test);
    let predicted: Vec<usize> = probs
        .rows()
        .into_iter()
        .map(|r| crate::svm::argmax(&r.to_vec()))
        .collect();
    let correct = predicted
        .iter()
        .zip(&y_test)
        .filter(|(p, a)| p == a)
        .count();
    println!(
        "\nTest accuracy: {:.2}% ({} of {})",
        correct as f64 / y_test.len() as f64 * 100.0,
        correct,
        y_test.len()
    );
    // Most common mistakes.
    let mut confusions = std::collections::HashMap::new();
    for (&p, &a) in predicted.iter().zip(&y_test) {
        if p != a {
            *confusions.entry((a, p)).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<((usize, usize), usize)> = confusions.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("\nMost common mistakes:");
    for ((a, p), n) in ranked.iter().take(5) {
        println!(
            "
a {a} read as {p}: {n} times"
        );
    }
    std::fs::write(DIGITS_MODEL, serde_json::to_string(&net)?)?;
    println!("\nSaved: {DIGITS_MODEL}");
    Ok(())
}
/// Show one test image and what the network makes of it.
pub fn show() -> AnyResult<()> {
    let index: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0".into())
        .parse()?;
    let net: Network = serde_json::from_str(&std::fs::read_to_string(DIGITS_MODEL)?)?;
    let (_, _, x_test, y_test) = load_mnist()?;
    let row = x_test.row(index);
    let probs = net.predict(&row.to_owned().insert_axis(Axis(0)));
    print!("{}", ascii(row));
    println!("Test image {index}: the real answer is {}", y_test[index]);
    let mut ranked: Vec<(usize, f64)> = probs.row(0).iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (digit, p) in ranked.iter().take(3) {
        println!("{digit}: {:.1}%", p * 100.0);
    }
    Ok(())
}
