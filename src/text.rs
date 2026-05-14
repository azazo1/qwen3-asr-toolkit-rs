use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

use crate::audio::{AudioChunk, WAV_SAMPLE_RATE};

pub fn post_text_process(text: &str, threshold: usize) -> String {
    let fixed_chars = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&fixed_chars, threshold, 20)
}

fn fix_char_repeats(input: &str, threshold: usize) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut result = String::new();
    let mut index = 0;

    while index < chars.len() {
        let mut count = 1;
        while index + count < chars.len() && chars[index + count] == chars[index] {
            count += 1;
        }

        if count > threshold {
            result.push(chars[index]);
        } else {
            for _ in 0..count {
                result.push(chars[index]);
            }
        }
        index += count;
    }

    result
}

fn fix_pattern_repeats(input: &str, threshold: usize, max_len: usize) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let min_repeat_chars = threshold * 2;
    if chars.len() < min_repeat_chars {
        return input.to_string();
    }

    let mut index = 0;
    let mut result = String::new();
    let mut found = false;

    while index <= chars.len().saturating_sub(min_repeat_chars) {
        let mut matched = false;
        for k in 1..=max_len {
            if index + k * threshold > chars.len() {
                break;
            }

            let pattern = &chars[index..index + k];
            let mut valid = true;
            for rep in 1..threshold {
                let start_idx = index + rep * k;
                if &chars[start_idx..start_idx + k] != pattern {
                    valid = false;
                    break;
                }
            }

            if valid {
                let mut end_index = index + threshold * k;
                while end_index + k <= chars.len() && &chars[end_index..end_index + k] == pattern {
                    end_index += k;
                }
                for ch in pattern {
                    result.push(*ch);
                }
                let tail = chars[end_index..].iter().collect::<String>();
                result.push_str(&fix_pattern_repeats(&tail, threshold, max_len));
                found = true;
                matched = true;
                index = chars.len();
                break;
            }
        }

        if matched {
            break;
        }

        result.push(chars[index]);
        index += 1;
    }

    if !found {
        result.push_str(&chars[index..].iter().collect::<String>());
    }

    result
}

pub fn write_srt(path: &Path, chunks: &[AudioChunk], results: &[(usize, String)]) -> Result<()> {
    let mut content = String::new();
    for (idx, (_, text)) in results.iter().enumerate() {
        let chunk = &chunks[idx];
        writeln!(&mut content, "{}", idx).expect("write to string");
        writeln!(
            &mut content,
            "{} --> {}",
            format_timestamp(chunk.start_sample),
            format_timestamp(chunk.end_sample)
        )
        .expect("write to string");
        writeln!(&mut content, "{}", text).expect("write to string");
        writeln!(&mut content).expect("write to string");
    }

    std::fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))
}

fn format_timestamp(sample: usize) -> String {
    let total_ms = (sample as f64 / WAV_SAMPLE_RATE as f64 * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let seconds = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_repeated_chars() {
        assert_eq!(post_text_process("aaaaaaaaaaaaaaaaaaaaa", 20), "a");
    }

    #[test]
    fn post_process_repeated_patterns() {
        assert_eq!(post_text_process("abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc", 20), "abc");
    }
}
