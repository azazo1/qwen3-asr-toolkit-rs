use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "qwen3-asr",
    version,
    about = "Rust toolkit for the Qwen3-ASR API - parallel high-throughput calls, robust long-audio transcription, multi-sample-rate support."
)]
pub struct Cli {
    #[arg(long = "input-file", short = 'i')]
    pub input_file: String,
    #[arg(long = "context", short = 'c', default_value = "")]
    pub context: String,
    #[arg(
        long = "language",
        short = 'l',
        help = "Manually specify a single language such as zh, en, Chinese, or English. Do not set this for mixed-language audio."
    )]
    pub language: Option<String>,
    #[arg(long = "dashscope-api-key")]
    pub dashscope_api_key: Option<String>,
    #[arg(long = "num-threads", short = 'j', default_value_t = 4)]
    pub num_threads: usize,
    #[arg(long = "vad-segment-threshold", short = 'd', default_value_t = 120)]
    pub vad_segment_threshold: usize,
    #[arg(long = "tmp-dir", short = 't')]
    pub tmp_dir: Option<PathBuf>,
    #[arg(long = "save-srt")]
    pub save_srt: bool,
    #[arg(long = "silence", short = 's')]
    pub silence: bool,
}

pub fn parse_args() -> Cli {
    let args = std::env::args().collect::<Vec<_>>();
    let mut rewritten = Vec::with_capacity(args.len());
    for arg in args {
        if arg == "-key" {
            rewritten.push("--dashscope-api-key".to_string());
        } else if arg == "-srt" {
            rewritten.push("--save-srt".to_string());
        } else {
            rewritten.push(arg);
        }
    }
    Cli::parse_from(rewritten)
}

impl Cli {
    pub fn validate(&self) -> Result<()> {
        if self.input_file.trim().is_empty() {
            bail!("--input-file is required");
        }
        if self.num_threads == 0 {
            bail!("--num-threads must be greater than 0");
        }
        if self.vad_segment_threshold == 0 {
            bail!("--vad-segment-threshold must be greater than 0");
        }
        Ok(())
    }

    pub fn tmp_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.tmp_dir {
            return Ok(path.clone());
        }

        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("failed to resolve home dir"))?;
        Ok(home.join("qwen3-asr-cache"))
    }
}
