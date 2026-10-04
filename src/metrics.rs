//! Scoring a list of probabilities against the real answers.
//! Works for any model that produces probabilities.
/// Counts of the four possible outcomes at one threshold.

pub struct Confusion {
    pub tp: usize,   // Said yes, was yes
    pub fp: usize,   // Said yes, was no
    pub tn: usize,   // Said no, was no
    pub fneg: usize, // Said no, was yes
}

impl Confusion {
    pub fn new(threshold: f64, probs: &[f64], actual: &[bool]) -> Self {
        let mut c = Confusion {
            tp: 0,
            fp: 0,
            tn: 0,
            fneg: 0,
        };
        for (&p, &a) in probs.iter().zip(actual) {
            match (p >= threshold, a) {
                (true, true) => c.tp += 1,
                (true, false) => c.fp += 1,
                (false, false) => c.tn += 1,
                (false, true) => c.fneg += 1,
            }
        }
        c
    }

    pub fn accuracy(&self) -> f64 {
        (self.tp + self.tn) as f64 / (self.tp + self.fp + self.tn + self.fneg) as f64
    }

    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }

    pub fn recall(&self) -> f64 {
        if self.tp + self.fneg == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fneg) as f64
        }
    }

    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

pub fn report(threshold: f64, probs: &[f64], actual: &[bool]) {
    let c = Confusion::new(threshold, probs, actual);
    println!("\n--- Threshold {threshold:.2} ---");
    println!("Accuracy: {:.2}%", c.accuracy() * 100.0);
    println!("Precision: {:.2}%", c.precision() * 100.0);
    println!("Recall:{:.2}%", c.recall() * 100.0);
    println!("F1:{:.3}", c.f1());
    println!("Predicted 0 Predicted 1");
    println!("Actual 0{:>11}{:>11}", c.tn, c.fp);
    println!("Actual 1{:>11}{:>11}", c.fneg, c.tp);
}

/// Chance that a random class-1 person gets a higher score
/// than a random class-0 person. 0.5 = guessing, 1.0 = perfect.

pub fn auc(probs: &[f64], actual: &[bool]) -> f64 {
    let pos: Vec<f64> = probs
        .iter()
        .zip(actual)
        .filter(|(_, a)| **a)
        .map(|(p, _)| *p)
        .collect();
    let neg: Vec<f64> = probs
        .iter()
        .zip(actual)
        .filter(|(_, a)| !**a)
        .map(|(p, _)| *p)
        .collect();

    let mut wins = 0.0;
    for p in &pos {
        for n in &neg {
            if p > n {
                wins += 1.0;
            } else if p == n {
                wins += 0.5
            }
        }
    }
    wins / (pos.len() * neg.len()) as f64
}
