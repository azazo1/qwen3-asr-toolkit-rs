use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub const WAV_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug)]
pub struct LoadedAudio {
    pub samples: Vec<f32>,
    pub file_name: String,
    pub file_stem: String,
}

impl LoadedAudio {
    pub fn duration_seconds(&self) -> f32 {
        self.samples.len() as f32 / WAV_SAMPLE_RATE as f32
    }
}

#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub start_sample: usize,
    pub end_sample: usize,
    pub samples: Vec<f32>,
}

pub async fn load_audio(file_path: &str) -> Result<LoadedAudio> {
    let input = file_path.to_string();
    tokio::task::spawn_blocking(move || load_audio_blocking(&input))
        .await
        .context("load_audio task join failed")?
}

pub async fn save_audio_file(wav: &[f32], file_path: &Path) -> Result<()> {
    let samples = wav.to_vec();
    let output = file_path.to_path_buf();
    tokio::task::spawn_blocking(move || save_audio_file_blocking(&samples, &output))
        .await
        .context("save_audio_file task join failed")?
}

fn load_audio_blocking(file_path: &str) -> Result<LoadedAudio> {
    let output = Command::new("ffmpeg")
        .arg("-i")
        .arg(file_path)
        .arg("-ar")
        .arg(WAV_SAMPLE_RATE.to_string())
        .arg("-ac")
        .arg("1")
        .arg("-f")
        .arg("s16le")
        .arg("-")
        .output()
        .with_context(|| format!("failed to run ffmpeg for {}", file_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("FFmpeg error processing '{}': {}", file_path, stderr.trim());
    }

    let samples = decode_pcm_s16le(&output.stdout)
        .context("failed to decode ffmpeg pcm output")?;

    let file_name = input_file_name(file_path);
    let file_stem = Path::new(&file_name)
        .file_stem()
        .unwrap_or_else(|| OsStr::new("audio"))
        .to_string_lossy()
        .into_owned();

    Ok(LoadedAudio {
        samples,
        file_name,
        file_stem,
    })
}

fn decode_pcm_s16le(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        bail!(
            "ffmpeg pcm output has odd byte length {}, expected 16-bit aligned data",
            bytes.len()
        );
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            let pcm = i16::from_le_bytes([chunk[0], chunk[1]]);
            pcm as f32 / i16::MAX as f32
        })
        .collect())
}

fn save_audio_file_blocking(wav: &[f32], file_path: &Path) -> Result<()> {
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: WAV_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(file_path, spec)
        .with_context(|| format!("failed to create {}", file_path.display()))?;

    for sample in wav {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32).round() as i16;
        writer
            .write_sample(pcm)
            .with_context(|| format!("failed to write {}", file_path.display()))?;
    }

    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", file_path.display()))
}

fn input_file_name(input: &str) -> String {
    if input.starts_with("http://") || input.starts_with("https://") {
        if let Ok(url) = url::Url::parse(input)
            && let Some(name) = Path::new(url.path()).file_name()
        {
            return name.to_string_lossy().into_owned();
        }
        return "remote_audio.wav".to_string();
    }

    PathBuf::from(input)
        .file_name()
        .unwrap_or_else(|| OsStr::new("audio.wav"))
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::decode_pcm_s16le;

    #[test]
    fn decode_pcm_s16le_reads_samples() {
        let bytes = [0_u8, 0_u8, 255_u8, 127_u8, 0_u8, 128_u8];
        let samples = decode_pcm_s16le(&bytes).expect("decoded pcm");
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], 0.0);
        assert!(samples[1] > 0.99);
        assert!(samples[2] < -0.99);
    }
}
