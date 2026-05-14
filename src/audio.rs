use std::ffi::OsStr;
use std::io::Cursor;
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
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg("-f")
        .arg("wav")
        .arg("-")
        .output()
        .with_context(|| format!("failed to run ffmpeg for {}", file_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("FFmpeg error processing '{}': {}", file_path, stderr.trim());
    }

    let cursor = Cursor::new(output.stdout);
    let mut reader = hound::WavReader::new(cursor).context("failed to decode ffmpeg wav output")?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != WAV_SAMPLE_RATE {
        bail!(
            "unexpected ffmpeg wav output: channels={}, sample_rate={}",
            spec.channels,
            spec.sample_rate
        );
    }

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read 16-bit pcm samples")?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read float pcm samples")?,
        _ => {
            bail!(
                "unsupported wav format from ffmpeg: {:?} {} bits",
                spec.sample_format,
                spec.bits_per_sample
            );
        }
    };

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
