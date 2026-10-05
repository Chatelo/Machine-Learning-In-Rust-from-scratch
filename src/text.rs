//! Chapter 7: turning tweets into numbers.
//!
//! Load the Coronavirus tweets, merge five sentiment labels into three,
//! split them fairly, break each tweet into words, and weight the words
//! with TF-IDF so a model can use them.

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::AnyResult;
use crate::data::{FOLDS, SEED, TEST_SHARE};

pub const RAW: &str = "data/Corona_NLP_train.csv";
pub const TRAIN: &str = "data/tweets_train.csv";
pub const TEST: &str = "data/tweets_test.csv";

pub const CLASSES: [&str; 3] = ["Negative", "Neutral", "Positive"];

/// A word must appear in at least this many training tweets to be kept.
const MIN_DOCS: usize = 2;

#[derive(Serialize, Deserialize)]
pub struct Tweet {
    pub text: String,
    pub label: usize, // index into CLASSES
}

// ---------- loading ----------
/// "Extremely Positive" becomes "Positive", and so on.
fn label_index(sentiment: &str) -> Option<usize> {
    match sentiment {
        "Extremely Negative" | "Negative" => Some(0),
        "Neutral" => Some(1),
        "Positive" | "Extremely Positive" => Some(2),
        _ => None,
    }
}

pub fn load_raw() -> AnyResult<Vec<Tweet>> {
    let mut reader = csv::Reader::from_path(RAW)?;
    let headers = reader.byte_headers()?.clone();
    let text_col = headers
        .iter()
        .position(|h| h == b"OriginalTweet")
        .ok_or("no OriginalTweet column")?;
    let label_col = headers
        .iter()
        .position(|h| h == b"Sentiment")
        .ok_or("no Sentiment column")?;
    let mut tweets = Vec::new();
    let mut repaired = 0;
    // Read raw bytes: a few tweets contain bytes that are not valid UTF-8.
    for record in reader.byte_records() {
        let record = record?;
        let raw_text = &record[text_col];
        let text = String::from_utf8_lossy(raw_text);
        if std::str::from_utf8(raw_text).is_err() {
            repaired += 1;
        }
        let sentiment = String::from_utf8_lossy(&record[label_col]);
        if let Some(label) = label_index(sentiment.trim()) {
            tweets.push(Tweet {
                text: text.into_owned(),
                label,
            });
        }
    }
    println!(
        "Loaded {} tweets ({repaired} had invalid bytes replaced)",
        tweets.len()
    );
    Ok(tweets)
}

pub fn load(path: &str) -> AnyResult<Vec<Tweet>> {
    let mut reader = csv::Reader::from_path(path)?;
    let tweets: Result<Vec<Tweet>, _> = reader.deserialize().collect();
    Ok(tweets?)
}
fn save(path: &str, tweets: &[&Tweet]) -> AnyResult<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for tweet in tweets {
        writer.serialize(tweet)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn print_balance(label: &str, tweets: &[Tweet]) {
    let mut counts = [0usize; 3];
    for t in tweets {
        counts[t.label] += 1;
    }
    let total = tweets.len() as f64;
    let parts: Vec<String> = CLASSES
        .iter()
        .zip(counts)
        .map(|(name, c)| format!("{name} {c} ({:.1}%)", c as f64 / total * 100.0))
        .collect();
    println!("{label}: {} | {}", tweets.len(), parts.join(" | "));
}

/// Shuffle each class separately and keep 20% of each for testing.
pub fn split() -> AnyResult<()> {
    let tweets = load_raw()?;
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for class in 0..CLASSES.len() {
        let mut rows: Vec<&Tweet> = tweets.iter().filter(|t| t.label == class).collect();
        rows.shuffle(&mut rng);
        let n_test = (rows.len() as f64 * TEST_SHARE).round() as usize;
        test.extend_from_slice(&rows[..n_test]);
        train.extend_from_slice(&rows[n_test..]);
    }
    train.shuffle(&mut rng);
    test.shuffle(&mut rng);
    save(TRAIN, &train)?;
    save(TEST, &test)?;
    print_balance("All", &tweets);
    print_balance("Train", &load(TRAIN)?);
    print_balance("Test ", &load(TEST)?);
    println!("Saved: {TRAIN}");
    println!("Saved: {TEST}");
    Ok(())
}

/// Give every tweet a fold number 0..FOLDS, class by class.
pub fn assign_folds(tweets: &[Tweet]) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut fold_of = vec![0; tweets.len()];
    for class in 0..CLASSES.len() {
        let mut rows: Vec<usize> = (0..tweets.len())
            .filter(|&i| tweets[i].label == class)
            .collect();
        rows.shuffle(&mut rng);
        for (k, &row) in rows.iter().enumerate() {
            fold_of[row] = k % FOLDS;
        }
    }
    fold_of
}

// ---------- words ----------
/// Lower-case, drop links and @usernames, keep runs of letters, digits and '.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|word| !word.starts_with("http") && !word.starts_with('@'))
        .flat_map(|word| {
            word.to_lowercase()
                .split(|c: char| !(c.is_alphanumeric() || c == '\''))
                .map(|w| w.trim_matches('\'').to_string())
                .collect::<Vec<_>>()
        })
        .filter(|w| w.chars().count() >= 2)
        .collect()
}

/// A sparse vector: only the words that appear, as (word index, weight).
pub type Sparse = Vec<(usize, f64)>;
/// The word list and each word's IDF weight, learned from training tweets only.
#[derive(Serialize, Deserialize, Clone)]
pub struct Vocabulary {
    pub words: Vec<String>,
    pub idf: Vec<f64>,
    #[serde(skip)]
    index: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn fit(texts: &[&str]) -> Self {
        // In how many tweets does each word appear?
        let mut doc_count: HashMap<String, usize> = HashMap::new();
        for text in texts {
            let mut seen: Vec<String> = tokenize(text);
            seen.sort();
            seen.dedup();
            for word in seen {
                *doc_count.entry(word).or_insert(0) += 1;
            }
        }
        let n = texts.len() as f64;
        let mut kept: Vec<(String, usize)> = doc_count
            .into_iter()
            .filter(|(_, c)| *c >= MIN_DOCS)
            .collect();
        kept.sort(); // alphabetical, so the word order is the same on every run
        let words: Vec<String> = kept.iter().map(|(w, _)| w.clone()).collect();
        // Rare words get a high weight, common words a low one.
        let idf: Vec<f64> = kept
            .iter()
            .map(|(_, c)| ((1.0 + n) / (1.0 + *c as f64)).ln() + 1.0)
            .collect();
        let mut vocab = Vocabulary {
            words,
            idf,
            index: HashMap::new(),
        };
        vocab.build_index();
        vocab
    }
    /// Rebuild the word-to-position lookup (needed after loading from JSON).
    pub fn build_index(&mut self) {
        self.index = self
            .words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
    }
    pub fn len(&self) -> usize {
        self.words.len()
    }
    /// TF-IDF vector for one text, scaled to length 1.
    pub fn transform(&self, text: &str) -> Sparse {
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for word in tokenize(text) {
            if let Some(&i) = self.index.get(&word) {
                *counts.entry(i).or_insert(0.0) += 1.0;
            }
        }
        let mut vector: Sparse = counts
            .into_iter()
            .map(|(i, tf)| (i, tf * self.idf[i]))
            .collect();
        vector.sort_by_key(|&(i, _)| i);
        let length = vector.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        if length > 0.0 {
            for (_, v) in vector.iter_mut() {
                *v /= length;
            }
        }
        vector
    }
}

// ---------- stage: look at the vocabulary ----------

pub fn vocab() -> AnyResult<()> {
    let train = load(TRAIN)?;
    let texts: Vec<&str> = train.iter().map(|t| t.text.as_str()).collect();
    let vocab = Vocabulary::fit(&texts);
    println!(
        "Vocabulary: {} words (each in at least {MIN_DOCS} training tweets)",
        vocab.len()
    );
    // Most common words = lowest IDF.
    let mut by_idf: Vec<usize> = (0..vocab.len()).collect();
    by_idf.sort_by(|&a, &b| vocab.idf[a].total_cmp(&vocab.idf[b]));
    let common: Vec<&str> = by_idf
        .iter()
        .take(15)
        .map(|&i| vocab.words[i].as_str())
        .collect();
    println!("Most common: {}", common.join(", "));
    let example = &train[0];
    println!(
        "\nExample tweet ({}):\n
{}",
        CLASSES[example.label], example.text
    );
    println!("Words: {:?}", tokenize(&example.text));
    let mut weights = vocab.transform(&example.text);
    weights.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("TF-IDF weights, highest first:");
    for (i, w) in weights.iter().take(8) {
        println!(
            "
{:<16} {w:.3}",
            vocab.words[*i]
        );
    }
    Ok(())
}
