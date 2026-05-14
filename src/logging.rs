use std::sync::{Arc, Once};

use indicatif::{ProgressBar, ProgressStyle};
use ort::logging::LogLevel;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();
static ORT_INIT: Once = Once::new();

pub fn init_tracing(silence: bool) {
    INIT.call_once(|| {
        let default_level = if silence { "warn" } else { "info" };
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_level));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_level(true)
            .init();
    });
}

pub fn init_ort_logging() {
    ORT_INIT.call_once(|| {
        let _ = ort::init()
            .with_logger(Arc::new(
                |level: LogLevel, category: &str, id: &str, code_location: &str, message: &str| {
                    let span = tracing::span!(
                        tracing::Level::TRACE,
                        "ort",
                        category = category,
                        id = id,
                        location = code_location
                    );
                    match level {
                        LogLevel::Verbose => {
                            tracing::event!(parent: &span, tracing::Level::TRACE, "{message}");
                        }
                        LogLevel::Info => {
                            tracing::event!(parent: &span, tracing::Level::DEBUG, "{message}");
                        }
                        LogLevel::Warning => {
                            tracing::event!(parent: &span, tracing::Level::WARN, "{message}");
                        }
                        LogLevel::Error | LogLevel::Fatal => {
                            tracing::event!(parent: &span, tracing::Level::ERROR, "{message}");
                        }
                    }
                }
            ))
            .commit();
    });
}

pub fn new_progress_bar(total: u64, silence: bool) -> ProgressBar {
    if silence {
        return ProgressBar::hidden();
    }

    let progress = ProgressBar::new(total);
    let style = ProgressStyle::with_template(
        "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
    )
    .expect("valid progress style");
    progress.set_style(style);
    progress.set_message("Calling Qwen3-ASR-Flash API");
    progress
}
