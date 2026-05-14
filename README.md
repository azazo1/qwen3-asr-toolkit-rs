# Qwen3-ASR-Toolkit Rust

这是一个基于 Rust 实现的 Qwen3-ASR 命令行工具, 功能来源于原始的 Python 项目 `/Users/azazo1/pjs/python/Qwen3-ASR-Toolkit`, 当前目标是保持原项目的核心能力和整体使用方式一致, 同时用 Rust 重写主要流程.

它面向长音频和长视频转写场景, 通过本地音频解码, VAD 分段, 并发调用 DashScope Qwen3-ASR API, 最终生成文本结果和可选的 SRT 字幕文件.

## 项目来源

这个项目不是从零设计的新工具, 而是对原 Python 项目的 Rust 迁移实现. 迁移时重点保留了下面这些能力:

- 兼容长于 3 分钟的音视频输入
- 使用 Silero VAD 按语音边界切分音频
- 多分片并发调用 Qwen3-ASR
- 对识别结果做重复片段清理
- 输出 `.txt` 文本和可选的 `.srt` 字幕
- 保留接近原项目的命令行参数风格

当前 Rust 版还在继续迭代, 但主流程已经完整可编译, 可运行, 并能独立完成与 Python 版相同的核心工作链路.

## 功能概览

- 支持本地音频和视频文件作为输入
- 自动调用 `ffmpeg` 将输入解码为 `16kHz` 单声道 PCM
- 对超过 3 分钟的音频执行 Silero VAD 分段
- 使用并发请求提升长音频整体转写速度
- 自动统计识别语种
- 自动生成文本输出文件
- 可选生成 SRT 字幕文件
- 支持上下文提示词和手动指定语种
- 支持静默模式和自定义临时目录

## 技术栈

这个 Rust 版的实现主要依赖下面这些技术和库:

- `tokio`
  用于异步运行时, 文件 IO, 异步任务调度
- `clap`
  用于命令行参数解析
- `reqwest`
  用于调用 DashScope HTTP API
- `serde` 和 `serde_json`
  用于请求和响应的序列化与反序列化
- `ort`
  用于加载并运行 Silero VAD 的 ONNX 模型
- `ndarray`
  用于构造 ONNX 推理输入输出张量
- `ffmpeg`
  作为外部依赖, 负责输入媒体解码与重采样
- `hound`
  用于将切分后的音频分片保存为 WAV 文件
- `indicatif`
  用于命令行进度条显示
- `tracing` 和 `tracing-subscriber`
  用于日志输出和调试信息分层

## 项目结构

当前代码结构比较直接, 每个模块都围绕主流程中的一个环节来组织:

- [src/main.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/main.rs)
  程序入口, 负责启动异步运行时
- [src/lib.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/lib.rs)
  主流程编排, 串联输入校验, 音频加载, VAD 分段, 并发请求, 输出写入
- [src/cli.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/cli.rs)
  命令行参数定义与兼容别名处理
- [src/audio.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/audio.rs)
  本地媒体解码, PCM 读取, 分片 WAV 保存
- [src/vad.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/vad.rs)
  Silero VAD 模型加载, 语音片段检测, 分段点计算
- [src/api.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/api.rs)
  DashScope Qwen3-ASR 请求构造, 重试逻辑, 响应解析
- [src/language.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/language.rs)
  语种别名归一化和识别结果统计
- [src/text.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/text.rs)
  文本重复清理和 SRT 生成
- [src/logging.rs](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/src/logging.rs)
  日志初始化, 进度条, ONNX Runtime 日志重映射
- [assets/silero_vad.onnx](/Users/azazo1/pjs/rust/qwen3-asr-toolkit/assets/silero_vad.onnx)
  本地 VAD 模型文件

## 整体流程

Rust 版整体执行流程如下:

1. 解析命令行参数并初始化日志系统
2. 检查输入路径或 URL 是否可访问
3. 使用 `ffmpeg` 将输入统一解码为 `16kHz` 单声道 PCM 流
4. 如果音频时长小于 180 秒, 直接作为单段处理
5. 如果音频时长超过 180 秒, 使用 Silero VAD 找出语音片段
6. 结合目标分段长度和最大分段长度计算最终切分点
7. 将每个分段保存为临时 WAV 文件
8. 并发调用 DashScope Qwen3-ASR API 识别各个分段
9. 收集结果后按原顺序拼接全文
10. 统计分段识别语种, 生成最终语种
11. 对识别文本执行重复字符和重复模式清理
12. 将最终结果写入 `.txt` 文件
13. 如果启用了 `--save-srt`, 额外生成 `.srt` 文件
14. 删除临时分片目录

## 命令行参数

当前支持的主要参数如下:

| 参数 | 说明 |
| --- | --- |
| `-i`, `--input-file` | 输入本地媒体文件路径 |
| `-c`, `--context` | 传给 ASR 的上下文文本 |
| `-l`, `--language` | 手动指定语种 |
| `--dashscope-api-key` | 指定 DashScope API Key |
| `-j`, `--num-threads` | 并发请求数 |
| `-d`, `--vad-segment-threshold` | VAD 目标分段长度, 单位秒 |
| `-t`, `--tmp-dir` | 临时分片目录 |
| `--save-srt` | 输出 SRT 字幕 |
| `-s`, `--silence` | 静默模式 |

为了兼容原 Python 项目, 当前还支持以下旧参数别名:

- `-key` 等价于 `--dashscope-api-key`
- `-srt` 等价于 `--save-srt`

## 使用示例

### 基本转写

```bash
qwen3-asr -i /path/to/input.mp4
```

### 指定 API Key

```bash
qwen3-asr -i /path/to/input.mp4 --dashscope-api-key sk-xxxx
```

### 指定上下文和语言

```bash
qwen3-asr -i /path/to/input.mp4 -c "Qwen3-ASR, DashScope, ONNX" -l zh
```

### 生成字幕

```bash
qwen3-asr -i /path/to/input.mp4 --save-srt
```

### 指定临时目录

```bash
qwen3-asr -i /path/to/input.mp4 -t /tmp/qwen3-asr-cache
```

### 查看调试日志

```bash
RUST_LOG=debug qwen3-asr -i /path/to/input.mp4
```

## 输出文件

默认情况下, 程序会在输入文件同目录生成:

- `xxx.txt`
- `xxx.srt`, 仅在启用 `--save-srt` 时生成

临时分片目录默认位于:

```text
~/qwen3-asr-cache
```

分片目录在流程结束后会自动清理.

## 运行前准备

使用前需要准备:

- Rust 工具链
- `ffmpeg`
- 可用的 DashScope API Key

建议将 API Key 放到环境变量中:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 当前实现说明

目前这个 Rust 版实现已经覆盖原 Python 项目的主干逻辑, 但仍有几个实现层面的特点需要注意:

- Silero VAD 模型当前以外部文件形式放在 `assets/` 目录中, 不是内嵌到可执行文件
- 默认日志只保留关键流程信息, ONNX Runtime 的内部 `info` 已经被压到 `debug`
- 音频解码走的是 `ffmpeg -> s16le PCM -> Rust 手动解码`, 避免了管道 WAV 头解析不稳定的问题
- 远程 URL 输入的头部可访问性检查已实现, 但日常验证主要还是围绕本地文件路径

## 开发与验证

当前开发过程中常用的验证命令:

```bash
RUSTC_WRAPPER= cargo clippy
RUSTC_WRAPPER= cargo test
RUSTC_WRAPPER= cargo run -- --help
```

## 后续可继续完善的方向

- 将 ONNX 模型内嵌到可执行文件
- 增加真正的端到端集成测试
- 进一步优化远程输入的处理方式
- 为输出和日志增加更细粒度的配置
