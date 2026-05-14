pub mod api;
pub mod audio;
pub mod cli;
pub mod language;
pub mod logging;
pub mod text;
pub mod vad;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use futures::stream::{self, StreamExt};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::api::QwenAsrClient;
use crate::audio::{AudioChunk, load_audio, save_audio_file};
use crate::cli::{Cli, parse_args};
use crate::language::majority_language;
use crate::logging::init_tracing;
use crate::text::write_srt;
use crate::vad::VadEngine;

const MAX_API_AUDIO_SECONDS: f32 = 180.0;

pub async fn run_cli() -> Result<()> {
    let cli = parse_args();
    init_tracing(cli.silence);
    run(cli).await
}

pub async fn run(cli: Cli) -> Result<()> {
    cli.validate()?;

    let api_key = cli
        .dashscope_api_key
        .clone()
        .or_else(|| std::env::var("DASHSCOPE_API_KEY").ok())
        .context("Please set DASHSCOPE_API_KEY or specify it with '-key'")?;

    let tmp_dir = cli.tmp_dir()?;
    fs::create_dir_all(&tmp_dir)
        .await
        .with_context(|| format!("failed to create tmp dir {}", tmp_dir.display()))?;

    let input = cli.input_file.clone();
    validate_input(&input).await?;

    let wav = load_audio(&input).await?;
    let duration_s = wav.duration_seconds();
    if !cli.silence {
        info!("Loaded wav duration: {:.2}s", duration_s);
    }

    let chunks = if duration_s >= MAX_API_AUDIO_SECONDS {
        if !cli.silence {
            info!("Wav duration is longer than 3 min, initializing Silero VAD model for segmenting...");
        }
        let vad = VadEngine::new()?;
        let segments = vad.process_vad(
            &wav.samples,
            cli.vad_segment_threshold,
            MAX_API_AUDIO_SECONDS as usize,
        )?;
        if !cli.silence {
            info!("Segmenting done, total segments: {}", segments.len());
        }
        segments
    } else {
        vec![AudioChunk {
            start_sample: 0,
            end_sample: wav.samples.len(),
            samples: wav.samples.clone(),
        }]
    };

    let save_dir = output_chunk_dir(&tmp_dir, &input);
    fs::create_dir_all(&save_dir)
        .await
        .with_context(|| format!("failed to create save dir {}", save_dir.display()))?;

    let mut chunk_paths = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk_path = save_dir.join(format!("{}_{}.wav", wav.file_stem, idx));
        save_audio_file(&chunk.samples, &chunk_path).await?;
        chunk_paths.push(chunk_path);
    }

    let client = Arc::new(QwenAsrClient::new(api_key));
    let progress = logging::new_progress_bar(chunks.len() as u64, cli.silence);

    let tasks = chunk_paths
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let client = Arc::clone(&client);
            let context = cli.context.clone();
            let language = cli.language.clone();
            let path = path.clone();
            async move {
                let result = client.transcribe_file(&path, &context, language.as_deref()).await;
                (idx, result)
            }
        })
        .collect::<Vec<_>>();

    let mut results = Vec::with_capacity(tasks.len());
    let mut languages = Vec::with_capacity(tasks.len());

    let mut stream = stream::iter(tasks).buffer_unordered(cli.num_threads);
    while let Some((idx, result)) = stream.next().await {
        let (language, text) = result?;
        progress.inc(1);
        results.push((idx, text));
        languages.push(language);
    }
    progress.finish_and_clear();

    results.sort_by_key(|(idx, _)| *idx);
    let full_text = results
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let detected_language = majority_language(&languages).unwrap_or_else(|| "Not Supported".to_string());

    if !cli.silence {
        info!("Detected Language: {}", detected_language);
        debug!("full transcription: {}", full_text);
    }

    let save_file = output_text_file(&input)?;
    let mut text_output = String::new();
    text_output.push_str(&detected_language);
    text_output.push('\n');
    text_output.push_str(&full_text);
    text_output.push('\n');
    fs::write(&save_file, text_output)
        .await
        .with_context(|| format!("failed to write {}", save_file.display()))?;

    println!(
        "Full transcription of \"{}\" from Qwen3-ASR-Flash API saved to \"{}\"!",
        input,
        save_file.display()
    );

    if cli.save_srt {
        let srt_file = save_file.with_extension("srt");
        write_srt(&srt_file, &chunks, &results)?;
        println!(
            "SRT subtitles of \"{}\" from Qwen3-ASR-Flash API saved to \"{}\"!",
            input,
            srt_file.display()
        );
    } else {
        println!(
            "SRT subtitles of \"{}\" from Qwen3-ASR-Flash API saved to \"{}\"!",
            input,
            save_dir.display()
        );
    }

    if let Err(error) = fs::remove_dir_all(&save_dir).await {
        warn!("failed to delete temp save dir {}: {}", save_dir.display(), error);
    }

    Ok(())
}

async fn validate_input(input: &str) -> Result<()> {
    if input.starts_with("http://") || input.starts_with("https://") {
        let client = reqwest::Client::new();
        let response = client
            .head(input)
            .send()
            .await
            .with_context(|| format!("HTTP link {} does not exist or is inaccessible", input))?;
        if response.status().as_u16() >= 400 {
            bail!(
                "HTTP link {} does not exist or is inaccessible: returned status code {}",
                input,
                response.status()
            );
        }
        return Ok(());
    }

    if !Path::new(input).exists() {
        bail!("Input file \"{}\" does not exist!", input);
    }

    Ok(())
}

fn output_chunk_dir(tmp_dir: &Path, input: &str) -> PathBuf {
    let wav_name = Path::new(input)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "remote_audio".to_string());
    let wav_dir_name = Path::new(&wav_name)
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".to_string());
    tmp_dir.join(wav_dir_name)
}

fn output_text_file(input: &str) -> Result<PathBuf> {
    if !input.starts_with("http://") && !input.starts_with("https://") {
        return Ok(Path::new(input).with_extension("txt"));
    }

    let url = url::Url::parse(input)?;
    let path = url.path().trim_matches('/');
    let filename = Path::new(path)
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "audio".to_string());
    Ok(PathBuf::from(format!("{}.txt", filename)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_text_file_local_path() {
        let path = output_text_file("/tmp/demo.mp3").expect("output path");
        assert_eq!(path, PathBuf::from("/tmp/demo.txt"));
    }

    #[test]
    fn output_text_file_remote_url() {
        let path = output_text_file("https://example.com/media/demo.mp3").expect("output path");
        assert_eq!(path, PathBuf::from("demo.txt"));
    }

    #[test]
    fn output_chunk_dir_uses_stem() {
        let path = output_chunk_dir(Path::new("/tmp/cache"), "/a/b/c.mp4");
        assert_eq!(path, PathBuf::from("/tmp/cache/c"));
    }
}
