use std::collections::BTreeSet;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::audio::{AudioChunk, WAV_SAMPLE_RATE};

const THRESHOLD: f32 = 0.5;
const NEG_THRESHOLD: f32 = 0.35;
const MIN_SPEECH_DURATION_MS: usize = 1500;
const MIN_SILENCE_DURATION_MS: usize = 500;
const WINDOW_SIZE_SAMPLES: usize = 512;
const CONTEXT_SIZE_SAMPLES: usize = 64;
const SILERO_VAD_MODEL_BYTES: &[u8] = include_bytes!("../assets/silero_vad.onnx");

pub struct VadEngine {
    model: Mutex<VadModel>,
}

impl VadEngine {
    pub fn new() -> Result<Self> {
        let mut builder = Session::builder()?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap_or_else(|error| error.recover());
        builder = builder
            .with_intra_threads(1)
            .unwrap_or_else(|error| error.recover());
        let session = builder.commit_from_memory(SILERO_VAD_MODEL_BYTES)?;

        Ok(Self {
            model: Mutex::new(VadModel::new(session)),
        })
    }

    pub fn process_vad(
        &self,
        wav: &[f32],
        segment_threshold_s: usize,
        max_segment_threshold_s: usize,
    ) -> Result<Vec<AudioChunk>> {
        match self.try_process_vad(wav, segment_threshold_s, max_segment_threshold_s) {
            Ok(chunks) => Ok(chunks),
            Err(_) => Ok(fallback_chunks(wav, max_segment_threshold_s)),
        }
    }

    fn try_process_vad(
        &self,
        wav: &[f32],
        segment_threshold_s: usize,
        max_segment_threshold_s: usize,
    ) -> Result<Vec<AudioChunk>> {
        let mut model = self.model.lock().expect("vad model mutex poisoned");
        let speech_timestamps = model.get_speech_timestamps(wav)?;
        if speech_timestamps.is_empty() {
            bail!("No speech segments detected by VAD.");
        }

        let mut potential_split_points = BTreeSet::new();
        potential_split_points.insert(0usize);
        potential_split_points.insert(wav.len());
        for timestamp in &speech_timestamps {
            potential_split_points.insert(timestamp.start);
        }
        let sorted_potential_splits = potential_split_points.into_iter().collect::<Vec<_>>();

        let mut final_split_points = BTreeSet::new();
        final_split_points.insert(0usize);
        final_split_points.insert(wav.len());

        let segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE as usize;
        let mut target_time = segment_threshold_samples;
        while target_time < wav.len() {
            let closest_point = sorted_potential_splits
                .iter()
                .min_by_key(|point| point.abs_diff(target_time))
                .copied()
                .context("failed to find closest split point")?;
            final_split_points.insert(closest_point);
            target_time += segment_threshold_samples;
        }

        let final_ordered_splits = final_split_points.into_iter().collect::<Vec<_>>();
        let max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE as usize;
        let mut refined_split_points = vec![0usize];

        for window in final_ordered_splits.windows(2) {
            let start = window[0];
            let end = window[1];
            let segment_length = end.saturating_sub(start);

            if segment_length <= max_segment_threshold_samples {
                refined_split_points.push(end);
                continue;
            }

            let num_subsegments = segment_length.div_ceil(max_segment_threshold_samples);
            let subsegment_length = segment_length as f64 / num_subsegments as f64;
            for index in 1..num_subsegments {
                let split_point = start + (index as f64 * subsegment_length).round() as usize;
                refined_split_points.push(split_point.min(end));
            }
            refined_split_points.push(end);
        }

        let mut chunks = Vec::new();
        for window in refined_split_points.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start || end > wav.len() {
                continue;
            }
            chunks.push(AudioChunk {
                start_sample: start,
                end_sample: end,
                samples: wav[start..end].to_vec(),
            });
        }

        Ok(chunks)
    }
}

#[derive(Clone, Copy, Debug)]
struct SpeechTimestamp {
    start: usize,
    end: usize,
}

struct VadModel {
    session: Session,
    state: Array3<f32>,
    context: Vec<f32>,
}

impl VadModel {
    fn new(session: Session) -> Self {
        Self {
            session,
            state: Array3::zeros((2, 1, 128)),
            context: vec![0.0; CONTEXT_SIZE_SAMPLES],
        }
    }

    fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    fn get_speech_timestamps(&mut self, audio: &[f32]) -> Result<Vec<SpeechTimestamp>> {
        self.reset_states();

        let min_speech_samples = WAV_SAMPLE_RATE as usize * MIN_SPEECH_DURATION_MS / 1000;
        let min_silence_samples = WAV_SAMPLE_RATE as usize * MIN_SILENCE_DURATION_MS / 1000;

        let mut speech_probs = Vec::new();
        let mut current_start = 0usize;
        while current_start < audio.len() {
            let mut chunk = vec![0.0f32; WINDOW_SIZE_SAMPLES];
            let chunk_len = (audio.len() - current_start).min(WINDOW_SIZE_SAMPLES);
            chunk[..chunk_len].copy_from_slice(&audio[current_start..current_start + chunk_len]);
            speech_probs.push(self.infer_chunk(&chunk)?);
            current_start += WINDOW_SIZE_SAMPLES;
        }

        let mut triggered = false;
        let mut speeches = Vec::new();
        let mut current_speech: Option<SpeechTimestamp> = None;
        let mut temp_end = 0usize;

        for (index, speech_prob) in speech_probs.iter().copied().enumerate() {
            if speech_prob >= THRESHOLD && temp_end != 0 {
                temp_end = 0;
            }

            if speech_prob >= THRESHOLD && !triggered {
                triggered = true;
                current_speech = Some(SpeechTimestamp {
                    start: WINDOW_SIZE_SAMPLES * index,
                    end: 0,
                });
                continue;
            }

            if speech_prob < NEG_THRESHOLD && triggered {
                if temp_end == 0 {
                    temp_end = WINDOW_SIZE_SAMPLES * index;
                }

                if WINDOW_SIZE_SAMPLES * index - temp_end < min_silence_samples {
                    continue;
                }

                if let Some(mut speech) = current_speech.take() {
                    speech.end = temp_end;
                    if speech.end.saturating_sub(speech.start) > min_speech_samples {
                        speeches.push(speech);
                    }
                }

                temp_end = 0;
                triggered = false;
            }
        }

        if triggered
            && let Some(mut speech) = current_speech.take()
        {
            speech.end = audio.len();
            if speech.end.saturating_sub(speech.start) > min_speech_samples {
                speeches.push(speech);
            }
        }

        Ok(speeches)
    }

    fn infer_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        if chunk.len() != WINDOW_SIZE_SAMPLES {
            bail!("vad chunk length must be {}", WINDOW_SIZE_SAMPLES);
        }

        let mut input_values = Vec::with_capacity(CONTEXT_SIZE_SAMPLES + WINDOW_SIZE_SAMPLES);
        input_values.extend_from_slice(&self.context);
        input_values.extend_from_slice(chunk);
        let input = Array2::from_shape_vec((1, CONTEXT_SIZE_SAMPLES + WINDOW_SIZE_SAMPLES), input_values)
            .context("failed to build vad input array")?;
        let sr = Array1::from_vec(vec![WAV_SAMPLE_RATE as i64]);
        let state = self.state.clone();

        let outputs = self.session.run(ort::inputs! {
            "input" => Tensor::from_array(input)?,
            "sr" => Tensor::from_array(sr)?,
            "state" => Tensor::from_array(state)?,
        })?;

        let output = outputs[0]
            .try_extract_array::<f32>()?
            .iter()
            .next()
            .copied()
            .context("vad output tensor is empty")?;
        let next_state = outputs[1]
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .context("failed to reshape vad state tensor")?;

        self.state = next_state;
        self.context.copy_from_slice(&chunk[WINDOW_SIZE_SAMPLES - CONTEXT_SIZE_SAMPLES..]);
        Ok(output)
    }
}

fn fallback_chunks(wav: &[f32], max_segment_threshold_s: usize) -> Vec<AudioChunk> {
    let mut chunks = Vec::new();
    let max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE as usize;
    let mut start = 0usize;

    while start < wav.len() {
        let end = (start + max_chunk_size_samples).min(wav.len());
        if end > start {
            chunks.push(AudioChunk {
                start_sample: start,
                end_sample: end,
                samples: wav[start..end].to_vec(),
            });
        }
        start = end;
    }

    chunks
}
