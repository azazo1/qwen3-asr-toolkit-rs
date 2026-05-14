use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use rand::prelude::RngExt;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::{debug, warn};

use crate::language::{code_to_language_name, normalize_language_code};
use crate::text::post_text_process;

const API_URL: &str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";
const MAX_API_RETRY: usize = 10;
const API_RETRY_SLEEP_MIN: f32 = 1.0;
const API_RETRY_SLEEP_MAX: f32 = 2.0;

#[derive(Clone)]
pub struct QwenAsrClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl QwenAsrClient {
    pub fn new(api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .expect("reqwest client");
        Self {
            api_key,
            model: "qwen3-asr-flash".to_string(),
            http,
        }
    }

    pub async fn transcribe_file(
        &self,
        audio_path: &Path,
        context: &str,
        language: Option<&str>,
    ) -> Result<(String, String)> {
        let normalized_language = normalize_language_code(language)?;
        let audio = local_audio_to_data_url(audio_path).await?;
        let mut last_error = String::new();

        for attempt in 0..MAX_API_RETRY {
            let request = DashScopeRequest::new(&self.model, context, &audio, normalized_language.as_deref());
            match self.send_request(&request).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = error.to_string();
                    if last_error.contains("DataInspectionFailed") {
                        warn!("data inspection failed for {}", audio_path.display());
                        break;
                    }
                    debug!(
                        "asr request failed on attempt {} for {}: {}",
                        attempt + 1,
                        audio_path.display(),
                        error
                    );
                }
            }

            if attempt + 1 < MAX_API_RETRY {
                let delay = rand::rng().random_range(API_RETRY_SLEEP_MIN..API_RETRY_SLEEP_MAX);
                sleep(Duration::from_secs_f32(delay)).await;
            }
        }

        bail!("{} task failed!\n{}", audio_path.display(), last_error)
    }

    async fn send_request(&self, request: &DashScopeRequest<'_>) -> Result<(String, String)> {
        let response = self
            .http
            .post(API_URL)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await
            .context("request to DashScope failed")?;

        let status = response.status();
        let body = response.text().await.context("failed to read DashScope response")?;

        if !status.is_success() {
            return Err(parse_api_error(status.as_u16(), &body));
        }

        let payload: DashScopeSuccessResponse =
            serde_json::from_str(&body).context("failed to parse DashScope success response")?;
        let choice = payload
            .output
            .choices
            .first()
            .context("missing output choice in DashScope response")?;

        let text = match &choice.message.content {
            ResponseContent::Items(items) => items.iter().map(|item| item.text.as_str()).collect::<String>(),
            ResponseContent::Plain(text) => text.clone(),
        };

        let language_code = choice
            .message
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.first())
            .map(|annotation| annotation.language.as_str());
        let language = code_to_language_name(language_code, normalized_language_name(request.parameters.asr_options.language).as_deref());
        let processed = post_text_process(&text, 20);
        Ok((language, processed))
    }
}

fn parse_api_error(status_code: u16, body: &str) -> anyhow::Error {
    if let Ok(error) = serde_json::from_str::<DashScopeErrorResponse>(body) {
        let code = error.code.unwrap_or_else(|| "UnknownError".to_string());
        let message = error.message.unwrap_or_else(|| body.to_string());
        return anyhow::anyhow!("http status_code: {} [{}] {}", status_code, code, message);
    }

    anyhow::anyhow!("http status_code: {} {}", status_code, body)
}

async fn local_audio_to_data_url(path: &Path) -> Result<String> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mime = mime_type_for_path(path);
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:{};base64,{}", mime, encoded))
}

fn mime_type_for_path(path: &Path) -> String {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("wav") => "audio/wav".to_string(),
        Some("mp3") => "audio/mpeg".to_string(),
        Some("m4a") => "audio/mp4".to_string(),
        Some("flac") => "audio/flac".to_string(),
        _ => mime_guess::from_path(path)
            .first_or_octet_stream()
            .essence_str()
            .to_string(),
    }
}

fn normalized_language_name(language: Option<&str>) -> Option<String> {
    language.map(|code| code_to_language_name(Some(code), None))
}

#[derive(Debug, Serialize)]
struct DashScopeRequest<'a> {
    model: &'a str,
    input: RequestInput<'a>,
    parameters: RequestParameters<'a>,
}

impl<'a> DashScopeRequest<'a> {
    fn new(model: &'a str, context: &'a str, audio: &'a str, language: Option<&'a str>) -> Self {
        Self {
            model,
            input: RequestInput {
                messages: vec![
                    RequestMessage {
                        role: "system",
                        content: vec![SystemContent { text: context }.into()],
                    },
                    RequestMessage {
                        role: "user",
                        content: vec![UserContent { audio }.into()],
                    },
                ],
            },
            parameters: RequestParameters {
                asr_options: AsrOptions {
                    enable_lid: true,
                    enable_itn: false,
                    language,
                },
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct RequestInput<'a> {
    messages: Vec<RequestMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct RequestMessage<'a> {
    role: &'a str,
    content: Vec<MessageContent<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum MessageContent<'a> {
    System(SystemContent<'a>),
    User(UserContent<'a>),
}

impl<'a> From<SystemContent<'a>> for MessageContent<'a> {
    fn from(value: SystemContent<'a>) -> Self {
        Self::System(value)
    }
}

impl<'a> From<UserContent<'a>> for MessageContent<'a> {
    fn from(value: UserContent<'a>) -> Self {
        Self::User(value)
    }
}

#[derive(Debug, Serialize)]
struct SystemContent<'a> {
    text: &'a str,
}

#[derive(Debug, Serialize)]
struct UserContent<'a> {
    audio: &'a str,
}

#[derive(Debug, Serialize)]
struct RequestParameters<'a> {
    asr_options: AsrOptions<'a>,
}

#[derive(Debug, Serialize)]
struct AsrOptions<'a> {
    enable_lid: bool,
    enable_itn: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct DashScopeSuccessResponse {
    output: SuccessOutput,
}

#[derive(Debug, Deserialize)]
struct SuccessOutput {
    choices: Vec<SuccessChoice>,
}

#[derive(Debug, Deserialize)]
struct SuccessChoice {
    message: SuccessMessage,
}

#[derive(Debug, Deserialize)]
struct SuccessMessage {
    content: ResponseContent,
    annotations: Option<Vec<AudioAnnotation>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponseContent {
    Items(Vec<TextContent>),
    Plain(String),
}

#[derive(Debug, Deserialize)]
struct TextContent {
    text: String,
}

#[derive(Debug, Deserialize)]
struct AudioAnnotation {
    language: String,
}

#[derive(Debug, Deserialize)]
struct DashScopeErrorResponse {
    code: Option<String>,
    message: Option<String>,
}
