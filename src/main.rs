use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    qwen3_asr_toolkit::run_cli().await
}
