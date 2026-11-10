//! Lexicon-constrained CTC scoring in the log domain, per §3.2.
//!
//! The model never decodes text freely. For each candidate string we build the extended
//! class sequence `0, c1, 0, c2, ..., cn, 0` and run the forward CTC recursion over the
//! softmax output, then accept the best candidate only when it clears a margin.

use std::cmp::Ordering;
use std::collections::HashMap;

/// Sentinel used in the `Vec<char>` class list; `index` never maps a real character to it.
pub const BLANK_CHAR: char = '\0';

/// `ln(max(p, 1e-12))`, computed in f64 from the model's f32 softmax.
#[inline]
pub fn log_prob(p: f32) -> f64 {
    (p as f64).max(1e-12).ln()
}

/// `ln(exp(a) + exp(b))`, stable at `-inf`.
#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Scores one fully extended label sequence over `t_used` timesteps of a `T x C` softmax.
pub fn score(probs: &[f32], c: usize, t_used: usize, ext: &[usize]) -> f64 {
    if t_used == 0 || ext.is_empty() {
        return f64::NEG_INFINITY;
    }
    let s = ext.len();
    let mut alpha = vec![f64::NEG_INFINITY; s];
    alpha[0] = log_prob(probs[ext[0]]);
    if s > 1 {
        alpha[1] = log_prob(probs[ext[1]]);
    }
    for t in 1..t_used {
        let base = t * c;
        let mut next = vec![f64::NEG_INFINITY; s];
        for k in (0..s).rev() {
            let mut a = alpha[k];
            if k >= 1 {
                a = log_sum_exp(a, alpha[k - 1]);
            }
            if k >= 2 && ext[k] != 0 && ext[k] != ext[k - 2] {
                a = log_sum_exp(a, alpha[k - 2]);
            }
            next[k] = a + log_prob(probs[base + ext[k]]);
        }
        alpha = next;
    }
    if s >= 2 {
        log_sum_exp(alpha[s - 1], alpha[s - 2])
    } else {
        alpha[s - 1]
    }
}

/// The extended class sequence for a word, or `None` when a character is not in the classes.
pub fn extension(word: &str, index: &HashMap<char, usize>) -> Option<Vec<usize>> {
    let mut ext = Vec::with_capacity(2 * word.chars().count() + 1);
    ext.push(0);
    for ch in word.chars() {
        let class = *index.get(&ch)?;
        ext.push(class);
        ext.push(0);
    }
    Some(ext)
}

/// The best-matching lexicon entry and its margin over the runner-up.
#[derive(Debug, Clone)]
pub struct Best {
    /// Index into the scored candidate list.
    pub index: usize,
    pub score: f64,
    /// `+inf` for a single-entry lexicon.
    pub margin: f64,
}

/// Scores every candidate and returns the best, breaking score ties by candidate text
/// descending.
///
/// With `case_variants` (names), the word is also scored upper- and lower-cased and the
/// maximum is kept; a variant with a character missing from the classes is skipped.
pub fn best(
    probs: &[f32],
    c: usize,
    t_used: usize,
    index: &HashMap<char, usize>,
    candidates: &[String],
    case_variants: bool,
) -> Option<Best> {
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (i, word) in candidates.iter().enumerate() {
        if let Some(s) = candidate_score(probs, c, t_used, index, word, case_variants) {
            scored.push((s, i));
        }
    }
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| candidates[b.1].cmp(&candidates[a.1]))
    });
    let margin = if scored.len() == 1 {
        f64::INFINITY
    } else {
        scored[0].0 - scored[1].0
    };
    Some(Best {
        index: scored[0].1,
        score: scored[0].0,
        margin,
    })
}

/// Best score for one candidate across its case variants.
fn candidate_score(
    probs: &[f32],
    c: usize,
    t_used: usize,
    index: &HashMap<char, usize>,
    word: &str,
    case_variants: bool,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut consider = |variant: &str| {
        if let Some(ext) = extension(variant, index) {
            let s = score(probs, c, t_used, &ext);
            if best.is_none_or(|b| s > b) {
                best = Some(s);
            }
        }
    };
    consider(word);
    if case_variants {
        consider(&word.to_uppercase());
        consider(&word.to_lowercase());
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes() -> HashMap<char, usize> {
        let mut index = HashMap::new();
        index.insert('a', 1);
        index.insert('b', 2);
        index.insert('5', 3);
        index
    }

    /// One timestep, uniform over blank and 'a': the word "a" scores `ln(0.5)`.
    #[test]
    fn single_timestep() {
        let probs = [0.5f32, 0.5, 0.0];
        let score = score(&probs, 3, 1, &[0, 1, 0]);
        assert!((score - 0.5f64.ln()).abs() < 1e-9);
    }

    /// "ab" cannot fit in one timestep; the recursion returns -inf.
    #[test]
    fn too_short_for_word() {
        let probs = [0.5f32, 0.25, 0.25];
        let score = score(&probs, 3, 1, &[0, 1, 0, 2, 0]);
        assert!(score == f64::NEG_INFINITY);
    }

    #[test]
    fn margin_is_infinite_for_one_entry() {
        let index = classes();
        let probs = vec![0.1f32, 0.6, 0.2, 0.1];
        let best = best(&probs, 4, 1, &index, &["a".to_string()], false).unwrap();
        assert_eq!(best.index, 0);
        assert!(best.margin.is_infinite());
    }

    #[test]
    fn unscorable_candidate_is_skipped() {
        let index = classes();
        let probs = vec![0.1f32, 0.6, 0.2, 0.1];
        // "z" is not in the classes; the single scorable entry wins with +inf margin.
        let best = best(
            &probs,
            4,
            1,
            &index,
            &["z".to_string(), "a".to_string()],
            false,
        )
        .unwrap();
        assert_eq!(best.index, 1);
        assert!(best.margin.is_infinite());
    }

    #[test]
    fn best_prefers_the_likelier_class_and_reports_margin() {
        let index = classes();
        // One timestep favouring 'a' (0.8) over 'b' (0.05); each single-char word can only
        // end on its character, so its score is that character's log-probability.
        let probs = vec![0.1f32, 0.8, 0.05, 0.05];
        let best = best(
            &probs,
            4,
            1,
            &index,
            &["b".to_string(), "a".to_string()],
            false,
        )
        .unwrap();
        assert_eq!(best.index, 1);
        assert!((best.margin - (0.8f64.ln() - 0.05f64.ln())).abs() < 1e-6);
    }
}
